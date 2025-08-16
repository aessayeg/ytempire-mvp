"""
Database configuration and session management for YTEmpire
Configured for 200 connections to handle 100+ videos/day
"""

from typing import AsyncGenerator
from sqlalchemy.ext.asyncio import AsyncSession, create_async_engine, async_sessionmaker
from sqlalchemy.orm import declarative_base
from sqlalchemy.pool import NullPool, QueuePool
import logging

from app.core.config import settings

logger = logging.getLogger(__name__)

# Determine if using SQLite or PostgreSQL
is_sqlite = not hasattr(settings, "DATABASE_URL") or "sqlite" in str(
    settings.DATABASE_URL
)

# Create async engine with proper pooling
if is_sqlite:
    # SQLite doesn't support connection pooling well
    engine = create_async_engine(
        settings.DATABASE_URL
        if hasattr(settings, "DATABASE_URL")
        else "sqlite+aiosqlite:///./ytempire.db",
        echo=False,
        poolclass=NullPool,  # NullPool for SQLite
    )
    logger.info("Using SQLite with NullPool")
else:
    # PostgreSQL with connection pooling for production
    engine = create_async_engine(
        settings.DATABASE_URL,
        echo=False,
        poolclass=QueuePool,
        pool_size=50,  # Base pool size
        max_overflow=150,  # Maximum overflow (total = 200 connections)
        pool_timeout=30,  # Timeout for getting connection
        pool_recycle=1800,  # Recycle connections after 30 minutes
        pool_pre_ping=True,  # Test connections before using
        echo_pool=False,  # Set to True for pool debugging
    )
    logger.info("Using PostgreSQL with QueuePool (200 max connections)")

# Create async session factory
AsyncSessionLocal = async_sessionmaker(
    engine,
    class_=AsyncSession,
    expire_on_commit=False,
    autocommit=False,
    autoflush=False,
)

# Create declarative base
Base = declarative_base()


async def get_session() -> AsyncGenerator[AsyncSession, None]:
    """
    Dependency to get database session
    """
    async with AsyncSessionLocal() as session:
        try:
            yield session
        finally:
            await session.close()


async def init_db():
    """
    Initialize database tables
    """
    async with engine.begin() as conn:
        # Import all models here to ensure they're registered
        from app.models import user, channel, video, analytics

        # Create all tables
        await conn.run_sync(Base.metadata.create_all)


async def close_db():
    """
    Close database connection
    """
    await engine.dispose()
