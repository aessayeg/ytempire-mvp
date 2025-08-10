"""
Database Configuration and Base Models
Owner: Backend Team Lead
"""

from sqlalchemy.ext.asyncio import AsyncSession, create_async_engine, async_sessionmaker
from sqlalchemy.orm import declarative_base
from app.core.config import settings

# Create async engine
engine = create_async_engine(
    settings.DATABASE_URL,
    echo=settings.DEBUG,
    future=True,
    pool_size=20,
    max_overflow=40,
    pool_pre_ping=True,
)

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

# Dependency to get DB session
async def get_db() -> AsyncSession:
    async with AsyncSessionLocal() as session:
        try:
            yield session
            await session.commit()
        except Exception:
            await session.rollback()
            raise
        finally:
            await session.close()


# Database health check
async def check_database_health() -> bool:
    """Check database connection health."""
    try:
        async with AsyncSessionLocal() as session:
            await session.execute("SELECT 1")
            return True
    except Exception:
        return False


# Initialize database
async def init_database():
    """Initialize database tables."""
    async with engine.begin() as conn:
        # Import all models to ensure they're registered
        from app.models.user import User
        from app.models.channel import Channel
        from app.models.video import Video
        from app.models.analytics import ChannelAnalytics, VideoAnalytics, CostTracking
        
        # Create all tables
        await conn.run_sync(Base.metadata.create_all)


# Close database connections
async def close_database():
    """Close database connections."""
    await engine.dispose()