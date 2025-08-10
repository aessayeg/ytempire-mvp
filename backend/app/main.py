"""
YTEmpire MVP - Main FastAPI Application
"""
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.trustedhost import TrustedHostMiddleware
from prometheus_fastapi_instrumentator import Instrumentator
import logging
from contextlib import asynccontextmanager

from app.core.config import settings
from app.api.v1.api import api_router
from app.core.logging import setup_logging
from app.db.session import engine
from app.db.base import Base

# Setup logging
setup_logging()
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Manage application lifecycle
    """
    # Startup
    logger.info("Starting YTEmpire API...")
    
    # Create database tables
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
    
    logger.info("Database initialized")
    logger.info(f"API running in {settings.ENVIRONMENT} mode")
    
    yield
    
    # Shutdown
    logger.info("Shutting down YTEmpire API...")
    await engine.dispose()


# Create FastAPI app
app = FastAPI(
    title=settings.PROJECT_NAME,
    version=settings.VERSION,
    openapi_url=f"{settings.API_V1_STR}/openapi.json",
    docs_url="/docs",
    redoc_url="/redoc",
    description="YTEmpire - AI-Powered YouTube Content Automation Platform",
    lifespan=lifespan
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.BACKEND_CORS_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Trusted host middleware
app.add_middleware(
    TrustedHostMiddleware,
    allowed_hosts=settings.ALLOWED_HOSTS
)

# Include API router
app.include_router(api_router, prefix=settings.API_V1_STR)

# Prometheus metrics
if settings.ENABLE_METRICS:
    Instrumentator().instrument(app).expose(app)

# Health check endpoint
@app.get("/health", tags=["Health"])
async def health_check():
    """
    Health check endpoint for monitoring
    """
    return {
        "status": "healthy",
        "version": settings.VERSION,
        "environment": settings.ENVIRONMENT
    }

# Root endpoint
@app.get("/", tags=["Root"])
async def root():
    """
    Root endpoint
    """
    return {
        "name": settings.PROJECT_NAME,
        "version": settings.VERSION,
        "description": "YTEmpire API - Automating YouTube Success",
        "documentation": "/docs"
    }