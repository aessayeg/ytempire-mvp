"""
YTEmpire Backend API - Main Application
Owner: Backend Team Lead
"""

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager
import uvicorn
import logging
from prometheus_client import make_asgi_app

from app.core.config import settings
from app.api.v1.api import api_router
from app.core.database import engine, Base, init_database, close_database
from app.middleware import (
    JWTAuthMiddleware,
    RateLimitMiddleware,
    RequestLoggingMiddleware
)
from app.core.metrics import PrometheusMiddleware, metrics

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup
    print("Starting YTEmpire Backend API...")
    
    # Create database tables
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
    
    # Initialize vector service
    try:
        from app.services.vector_service import vector_service
        await vector_service.initialize()
        logger.info("Vector service initialized successfully")
    except Exception as e:
        logger.warning(f"Failed to initialize vector service: {str(e)}")
        logger.warning("Vector-based features will not be available")
    
    # Initialize secrets manager
    try:
        from app.services.secrets_manager import secrets_manager
        await secrets_manager.initialize()
        logger.info("Secrets manager initialized successfully")
    except Exception as e:
        logger.warning(f"Failed to initialize secrets manager: {str(e)}")
        logger.warning("Advanced secrets management features will not be available")
    
    yield
    
    # Shutdown
    print("Shutting down YTEmpire Backend API...")
    await engine.dispose()


app = FastAPI(
    title=settings.PROJECT_NAME,
    version=settings.VERSION,
    openapi_url=f"{settings.API_V1_STR}/openapi.json",
    docs_url="/docs",
    redoc_url="/redoc",
    lifespan=lifespan
)

# Add custom middleware (order matters - add in reverse execution order)
app.add_middleware(RequestLoggingMiddleware)
app.add_middleware(JWTAuthMiddleware)
app.add_middleware(RateLimitMiddleware, requests_per_minute=60)
app.add_middleware(PrometheusMiddleware)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.BACKEND_CORS_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include API router
app.include_router(api_router, prefix=settings.API_V1_STR)

# Add prometheus metrics endpoint
metrics_app = make_asgi_app()
app.mount("/metrics", metrics_app)

# Health check endpoint
@app.get("/health")
async def health_check():
    """Comprehensive health check endpoint for monitoring."""
    from datetime import datetime
    from app.core.database import AsyncSessionLocal
    
    health_status = {
        "status": "healthy",
        "service": "ytempire-backend",
        "version": settings.VERSION,
        "timestamp": datetime.utcnow().isoformat(),
        "checks": {
            "database": {"status": "unknown"},
            "redis": {"status": "unknown"},
            "vector_db": {"status": "unknown"}
        }
    }
    
    # Test database connection
    try:
        async with AsyncSessionLocal() as db:
            await db.execute("SELECT 1")
        health_status["checks"]["database"]["status"] = "healthy"
    except Exception as e:
        health_status["checks"]["database"]["status"] = "unhealthy"
        health_status["checks"]["database"]["error"] = str(e)
        health_status["status"] = "degraded"
    
    # Test Redis connection
    try:
        import redis.asyncio as redis
        redis_client = redis.from_url(f"redis://{settings.REDIS_HOST}:6379")
        await redis_client.ping()
        health_status["checks"]["redis"]["status"] = "healthy"
        await redis_client.close()
    except Exception as e:
        health_status["checks"]["redis"]["status"] = "unhealthy"
        health_status["checks"]["redis"]["error"] = str(e)
        health_status["status"] = "degraded"
    
    # Test Vector Database connection
    try:
        from app.services.vector_service import vector_service
        if vector_service.initialized:
            # Test with a simple query
            stats = await vector_service.get_collection_stats()
            if stats:
                health_status["checks"]["vector_db"]["status"] = "healthy"
                health_status["checks"]["vector_db"]["collections"] = len(stats)
            else:
                health_status["checks"]["vector_db"]["status"] = "degraded"
        else:
            health_status["checks"]["vector_db"]["status"] = "not_initialized"
    except Exception as e:
        health_status["checks"]["vector_db"]["status"] = "unhealthy"
        health_status["checks"]["vector_db"]["error"] = str(e)
        health_status["status"] = "degraded"
    
    return health_status

@app.get("/health/ready")
async def readiness_check():
    """Kubernetes readiness probe endpoint."""
    from app.core.database import AsyncSessionLocal
    
    try:
        async with AsyncSessionLocal() as db:
            await db.execute("SELECT 1")
        return {"status": "ready"}
    except Exception:
        from fastapi import HTTPException, status
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Service not ready"
        )

@app.get("/health/live")
async def liveness_check():
    """Kubernetes liveness probe endpoint."""
    return {"status": "alive"}

@app.get("/")
async def root():
    return {
        "message": "YTEmpire Backend API",
        "version": settings.VERSION,
        "docs": "/docs"
    }


if __name__ == "__main__":
    uvicorn.run(
        "app.main:app",
        host="0.0.0.0",
        port=8000,
        reload=settings.DEBUG
    )