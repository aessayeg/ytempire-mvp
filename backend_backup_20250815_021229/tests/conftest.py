"""
Pytest configuration and fixtures for YTEmpire backend tests
"""
import pytest
import asyncio
from typing import Generator, AsyncGenerator
from fastapi.testclient import TestClient
from sqlalchemy.ext.asyncio import AsyncSession, create_async_engine, async_sessionmaker
from sqlalchemy.pool import NullPool
import redis.asyncio as redis

from app.main import app
from app.db.session import get_db
from app.db.base import Base
from app.core.config import settings
from app.core.security import create_access_token
from app.models.user import User

# Test database URL
TEST_DATABASE_URL = "postgresql+asyncpg://test_user:test_pass@localhost:5433/test_db"
TEST_REDIS_URL = "redis://localhost:6380/0"


@pytest.fixture(scope="session")
def event_loop():
    """Create an instance of the default event loop for the test session."""
    loop = asyncio.get_event_loop_policy().new_event_loop()
    yield loop
    loop.close()


@pytest.fixture(scope="function")
async def async_engine():
    """Create async engine for testing"""
    engine = create_async_engine(
        TEST_DATABASE_URL,
        poolclass=NullPool,
        echo=False,
    )
    
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
    
    yield engine
    
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.drop_all)
    
    await engine.dispose()


@pytest.fixture(scope="function")
async def async_session(async_engine) -> AsyncGenerator[AsyncSession, None]:
    """Create async session for testing"""
    async_session_maker = async_sessionmaker(
        async_engine,
        class_=AsyncSession,
        expire_on_commit=False,
    )
    
    async with async_session_maker() as session:
        yield session


@pytest.fixture(scope="function")
async def redis_client():
    """Create Redis client for testing"""
    client = await redis.from_url(TEST_REDIS_URL, decode_responses=True)
    yield client
    await client.flushdb()
    await client.close()


@pytest.fixture(scope="function")
def client(async_session) -> Generator:
    """Create test client with overridden dependencies"""
    
    async def override_get_db():
        yield async_session
    
    app.dependency_overrides[get_db] = override_get_db
    
    with TestClient(app) as test_client:
        yield test_client
    
    app.dependency_overrides.clear()


@pytest.fixture
async def test_user(async_session: AsyncSession) -> User:
    """Create a test user"""
    user = User(
        email="test@example.com",
        username="testuser",
        hashed_password="$2b$12$EixZaYVK1fsbw1ZfbX3OXePaWxn96p36WQoeG6Lruj3vjPGga31lW",  # secret
        full_name="Test User",
        is_active=True,
        is_verified=True,
        subscription_tier="free",
        channels_limit=1,
        videos_per_day_limit=3,
    )
    async_session.add(user)
    await async_session.commit()
    await async_session.refresh(user)
    return user


@pytest.fixture
async def test_pro_user(async_session: AsyncSession) -> User:
    """Create a test pro user"""
    user = User(
        email="pro@example.com",
        username="prouser",
        hashed_password="$2b$12$EixZaYVK1fsbw1ZfbX3OXePaWxn96p36WQoeG6Lruj3vjPGga31lW",  # secret
        full_name="Pro User",
        is_active=True,
        is_verified=True,
        subscription_tier="pro",
        channels_limit=5,
        videos_per_day_limit=10,
    )
    async_session.add(user)
    await async_session.commit()
    await async_session.refresh(user)
    return user


@pytest.fixture
async def test_admin_user(async_session: AsyncSession) -> User:
    """Create a test admin user"""
    user = User(
        email="admin@example.com",
        username="adminuser",
        hashed_password="$2b$12$EixZaYVK1fsbw1ZfbX3OXePaWxn96p36WQoeG6Lruj3vjPGga31lW",  # secret
        full_name="Admin User",
        is_active=True,
        is_verified=True,
        is_superuser=True,
        subscription_tier="enterprise",
        channels_limit=999,
        videos_per_day_limit=999,
    )
    async_session.add(user)
    await async_session.commit()
    await async_session.refresh(user)
    return user


@pytest.fixture
def auth_headers(test_user: User) -> dict:
    """Create authentication headers for test user"""
    access_token = create_access_token(subject=str(test_user.id))
    return {"Authorization": f"Bearer {access_token}"}


@pytest.fixture
def pro_auth_headers(test_pro_user: User) -> dict:
    """Create authentication headers for pro user"""
    access_token = create_access_token(subject=str(test_pro_user.id))
    return {"Authorization": f"Bearer {access_token}"}


@pytest.fixture
def admin_auth_headers(test_admin_user: User) -> dict:
    """Create authentication headers for admin user"""
    access_token = create_access_token(subject=str(test_admin_user.id))
    return {"Authorization": f"Bearer {access_token}"}