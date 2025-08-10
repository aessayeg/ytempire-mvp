"""
Test Configuration and Fixtures
Owner: QA Engineer #1
"""

import asyncio
import pytest
import pytest_asyncio
from typing import AsyncGenerator, Generator
from sqlalchemy.ext.asyncio import AsyncSession, create_async_engine
from sqlalchemy.orm import sessionmaker
from fastapi.testclient import TestClient
from httpx import AsyncClient
import redis.asyncio as redis

from app.main import app
from app.core.database import Base, get_db
from app.core.config import settings
from app.models.user import User
from app.models.channel import Channel
from app.models.video import Video
from app.services.auth_service import AuthService
from app.repositories.user_repository import UserRepository

# Test database engine
TEST_DATABASE_URL = "postgresql+asyncpg://test:test@localhost/test_ytempire_db"
test_engine = create_async_engine(
    TEST_DATABASE_URL,
    echo=False,
    future=True
)

TestSessionLocal = sessionmaker(
    test_engine,
    class_=AsyncSession,
    expire_on_commit=False
)


@pytest.fixture(scope="session")
def event_loop() -> Generator[asyncio.AbstractEventLoop, None, None]:
    """Create an instance of the default event loop for the test session."""
    loop = asyncio.new_event_loop()
    yield loop
    loop.close()


@pytest_asyncio.fixture(scope="function")
async def db_session() -> AsyncGenerator[AsyncSession, None]:
    """Create test database session."""
    async with test_engine.begin() as connection:
        # Create all tables
        await connection.run_sync(Base.metadata.drop_all)
        await connection.run_sync(Base.metadata.create_all)
        
        # Create session
        async with TestSessionLocal(bind=connection) as session:
            yield session
            await session.rollback()


@pytest.fixture
def client(db_session: AsyncSession) -> TestClient:
    """Create test client with database session override."""
    
    def override_get_db():
        try:
            yield db_session
        finally:
            pass
    
    app.dependency_overrides[get_db] = override_get_db
    
    with TestClient(app) as test_client:
        yield test_client
    
    app.dependency_overrides.clear()


@pytest_asyncio.fixture
async def async_client(db_session: AsyncSession) -> AsyncGenerator[AsyncClient, None]:
    """Create async test client."""
    
    def override_get_db():
        try:
            yield db_session
        finally:
            pass
    
    app.dependency_overrides[get_db] = override_get_db
    
    async with AsyncClient(app=app, base_url="http://test") as client:
        yield client
    
    app.dependency_overrides.clear()


@pytest_asyncio.fixture
async def redis_client() -> AsyncGenerator[redis.Redis, None]:
    """Create Redis client for testing."""
    client = redis.Redis.from_url("redis://localhost:6379/1", decode_responses=True)
    
    # Clear test database
    await client.flushdb()
    
    yield client
    
    # Cleanup
    await client.flushdb()
    await client.close()


@pytest_asyncio.fixture
async def user_repo(db_session: AsyncSession) -> UserRepository:
    """Create user repository for testing."""
    return UserRepository(db_session)


@pytest_asyncio.fixture
async def auth_service(user_repo: UserRepository) -> AuthService:
    """Create auth service for testing."""
    return AuthService(user_repo)


@pytest_asyncio.fixture
async def test_user(db_session: AsyncSession, user_repo: UserRepository) -> User:
    """Create test user."""
    from app.schemas.auth import UserRegister
    from app.services.auth_service import AuthService
    
    auth_service = AuthService(user_repo)
    
    user_data = UserRegister(
        email="test@example.com",
        username="testuser",
        password="testpassword123",
        full_name="Test User"
    )
    
    user = await auth_service.register_user(user_data)
    return user


@pytest_asyncio.fixture
async def authenticated_user(test_user: User, auth_service: AuthService) -> dict:
    """Create authenticated user with token."""
    token = await auth_service.login_user("test@example.com", "testpassword123")
    return {
        "user": test_user,
        "token": token
    }


@pytest_asyncio.fixture
async def test_channel(db_session: AsyncSession, test_user: User) -> Channel:
    """Create test channel."""
    from app.models.channel import Channel
    import uuid
    from datetime import datetime
    
    channel = Channel(
        id=str(uuid.uuid4()),
        user_id=test_user.id,
        name="Test Channel",
        description="A test channel",
        category="technology",
        target_audience="developers",
        content_style="educational",
        is_active=True,
        created_at=datetime.utcnow(),
        updated_at=datetime.utcnow()
    )
    
    db_session.add(channel)
    await db_session.commit()
    await db_session.refresh(channel)
    
    return channel


@pytest_asyncio.fixture
async def test_video(db_session: AsyncSession, test_user: User, test_channel: Channel) -> Video:
    """Create test video."""
    from app.models.video import Video, VideoStatus
    import uuid
    from datetime import datetime
    
    video = Video(
        id=str(uuid.uuid4()),
        channel_id=test_channel.id,
        user_id=test_user.id,
        title="Test Video",
        description="A test video",
        status=VideoStatus.PENDING,
        priority=1,
        created_at=datetime.utcnow(),
        updated_at=datetime.utcnow()
    )
    
    db_session.add(video)
    await db_session.commit()
    await db_session.refresh(video)
    
    return video


@pytest.fixture
def mock_youtube_api():
    """Mock YouTube API responses."""
    import unittest.mock
    
    class MockYouTubeAPI:
        def __init__(self):
            self.videos = {}
            self.channels = {}
        
        def upload_video(self, title, description, file_path):
            video_id = f"mock_video_{len(self.videos) + 1}"
            self.videos[video_id] = {
                "id": video_id,
                "title": title,
                "description": description,
                "status": "uploaded"
            }
            return {"id": video_id}
        
        def get_video_stats(self, video_id):
            if video_id in self.videos:
                return {
                    "views": 100,
                    "likes": 10,
                    "comments": 5
                }
            return None
    
    with unittest.mock.patch('app.services.youtube_service.YouTubeService') as mock:
        mock.return_value = MockYouTubeAPI()
        yield mock


@pytest.fixture
def mock_openai_api():
    """Mock OpenAI API responses."""
    import unittest.mock
    
    def mock_completion(*args, **kwargs):
        return {
            "choices": [{
                "message": {
                    "content": "This is a mock AI response for testing purposes."
                }
            }],
            "usage": {
                "total_tokens": 100
            }
        }
    
    with unittest.mock.patch('openai.ChatCompletion.create', side_effect=mock_completion):
        yield


@pytest.fixture
def mock_elevenlabs_api():
    """Mock ElevenLabs API responses."""
    import unittest.mock
    
    def mock_generate(*args, **kwargs):
        # Return mock audio bytes
        return b"mock_audio_data"
    
    with unittest.mock.patch('elevenlabs.generate', side_effect=mock_generate):
        yield


# Pytest configuration
def pytest_configure(config):
    """Configure pytest with custom markers."""
    config.addinivalue_line("markers", "unit: Unit tests")
    config.addinivalue_line("markers", "integration: Integration tests")
    config.addinivalue_line("markers", "e2e: End-to-end tests")
    config.addinivalue_line("markers", "slow: Slow tests")
    config.addinivalue_line("markers", "auth: Authentication tests")
    config.addinivalue_line("markers", "api: API tests")
    config.addinivalue_line("markers", "db: Database tests")


def pytest_collection_modifyitems(config, items):
    """Modify test collection to add markers based on test location."""
    for item in items:
        # Add markers based on test file location
        if "unit" in str(item.fspath):
            item.add_marker(pytest.mark.unit)
        elif "integration" in str(item.fspath):
            item.add_marker(pytest.mark.integration)
        elif "e2e" in str(item.fspath):
            item.add_marker(pytest.mark.e2e)
        
        # Add auth marker for auth tests
        if "auth" in str(item.fspath):
            item.add_marker(pytest.mark.auth)
        
        # Add api marker for API tests
        if "api" in str(item.fspath):
            item.add_marker(pytest.mark.api)
        
        # Add db marker for database tests
        if any(keyword in str(item.fspath) for keyword in ["repository", "model", "database"]):
            item.add_marker(pytest.mark.db)