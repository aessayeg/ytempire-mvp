# YTEMPIRE API Implementation Technical Guide

**Document Version**: 1.0  
**Date**: January 2025  
**For**: API Development Engineer  
**Classification**: Technical Implementation Reference

---

## 📚 Table of Contents

1. [Executive Summary](#1-executive-summary)
2. [System Architecture Overview](#2-system-architecture-overview)
3. [Development Environment Setup](#3-development-environment-setup)
4. [Project Structure](#4-project-structure)
5. [Core Technologies & Dependencies](#5-core-technologies--dependencies)
6. [Database Design & Models](#6-database-design--models)
7. [Authentication & Security](#7-authentication--security)
8. [API Endpoints Implementation](#8-api-endpoints-implementation)
9. [YouTube Multi-Account Integration](#9-youtube-multi-account-integration)
10. [Cost Tracking System](#10-cost-tracking-system)
11. [N8N Webhook Integration](#11-n8n-webhook-integration)
12. [Performance Optimization](#12-performance-optimization)
13. [Testing Strategy](#13-testing-strategy)
14. [Deployment Guide](#14-deployment-guide)
15. [Monitoring & Logging](#15-monitoring--logging)

---

## 1. Executive Summary

### Your Role
As the API Development Engineer for YTEMPIRE, you're responsible for building the backend APIs that power an automated YouTube content empire platform. The system must handle 50+ videos daily, manage 500+ channels, and maintain <$3/video costs while achieving 95% automation.

### Key Responsibilities
- **API Development**: Build RESTful APIs using FastAPI
- **YouTube Integration**: Manage 15 YouTube accounts with quota optimization
- **Cost Tracking**: Real-time cost monitoring with hard limits
- **Performance**: Ensure <500ms p95 response times
- **Reliability**: Maintain 99.9% uptime SLA

### Critical Success Metrics
```python
KEY_METRICS = {
    "daily_videos": 50,              # MVP target
    "concurrent_users": 100,         # System capacity
    "channels_managed": 500,         # Total channels across all users
    "cost_per_video": 3.00,         # Maximum USD
    "api_response_p95": 500,        # Milliseconds
    "automation_rate": 95,          # Percentage
    "youtube_accounts": 15,         # 12 active + 3 reserve
    "uptime_target": 99.9           # Percentage
}
```

---

## 2. System Architecture Overview

### High-Level Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                         Frontend (React)                      │
└─────────────────────────────────────────────────────────────┘
                                │
                                ▼
┌─────────────────────────────────────────────────────────────┐
│                      API Gateway (FastAPI)                    │
│                    ┌─────────────────────┐                   │
│                    │   Authentication    │                   │
│                    │   Rate Limiting     │                   │
│                    │   Request Routing   │                   │
│                    └─────────────────────┘                   │
└─────────────────────────────────────────────────────────────┘
                                │
        ┌───────────────────────┼───────────────────────┐
        ▼                       ▼                       ▼
┌──────────────┐     ┌──────────────┐      ┌──────────────┐
│   Core API   │     │ YouTube API  │      │  Cost API    │
│   Services   │     │   Service    │      │   Service    │
└──────────────┘     └──────────────┘      └──────────────┘
        │                    │                      │
        └────────────────────┼──────────────────────┘
                             ▼
        ┌────────────────────────────────────────────┐
        │           PostgreSQL Database              │
        │                                            │
        │  ┌──────────┐  ┌──────────┐  ┌─────────┐ │
        │  │  Users   │  │ Channels │  │ Videos  │ │
        │  └──────────┘  └──────────┘  └─────────┘ │
        └────────────────────────────────────────────┘
                             │
                             ▼
        ┌────────────────────────────────────────────┐
        │              Redis Cache                   │
        │         (Session, Rate Limit, Cost)        │
        └────────────────────────────────────────────┘
```

### Component Responsibilities

| Component | Responsibility | Technologies |
|-----------|---------------|--------------|
| API Gateway | Request routing, authentication, rate limiting | FastAPI, JWT |
| Core Services | Business logic, CRUD operations | Python, SQLAlchemy |
| YouTube Service | Multi-account management, uploads | Google API Client |
| Cost Service | Real-time tracking, budget enforcement | Redis, PostgreSQL |
| Database | Data persistence, transactions | PostgreSQL 14+ |
| Cache | Session storage, rate limiting, real-time metrics | Redis 7+ |

---

## 3. Development Environment Setup

### Prerequisites

```bash
# System Requirements
- Ubuntu 22.04 / macOS 13+ / Windows 11 with WSL2
- Python 3.11+
- PostgreSQL 14+
- Redis 7+
- Node.js 18+ (for N8N)
- 16GB RAM minimum
- 50GB free disk space
```

### Initial Setup

```bash
# 1. Clone repository
git clone https://github.com/ytempire/backend.git
cd backend

# 2. Create Python virtual environment
python3.11 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# 3. Install dependencies
pip install --upgrade pip
pip install -r requirements.txt
pip install -r requirements-dev.txt  # Development dependencies

# 4. Setup PostgreSQL
sudo -u postgres psql
CREATE DATABASE ytempire_db;
CREATE USER ytempire_user WITH ENCRYPTED PASSWORD 'secure_password';
GRANT ALL PRIVILEGES ON DATABASE ytempire_db TO ytempire_user;
\q

# 5. Setup Redis
sudo systemctl start redis-server
redis-cli ping  # Should return PONG

# 6. Environment configuration
cp .env.example .env
# Edit .env file with your configuration

# 7. Run database migrations
alembic upgrade head

# 8. Start development server
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

### Environment Variables (.env)

```bash
# Application
PROJECT_NAME=YTEMPIRE
ENVIRONMENT=development
SECRET_KEY=your-secret-key-here-minimum-32-chars
API_V1_STR=/api/v1

# Database
POSTGRES_SERVER=localhost
POSTGRES_USER=ytempire_user
POSTGRES_PASSWORD=secure_password
POSTGRES_DB=ytempire_db
POSTGRES_PORT=5432

# Redis
REDIS_HOST=localhost
REDIS_PORT=6379
REDIS_DB=0

# YouTube API (from Google Cloud Console)
YOUTUBE_CLIENT_ID=your-client-id.apps.googleusercontent.com
YOUTUBE_CLIENT_SECRET=your-client-secret
YOUTUBE_REDIRECT_URI=http://localhost:8000/api/v1/auth/youtube/callback

# OpenAI
OPENAI_API_KEY=sk-your-openai-api-key

# Stripe
STRIPE_SECRET_KEY=sk_test_your-stripe-key
STRIPE_WEBHOOK_SECRET=whsec_your-webhook-secret

# N8N Integration
N8N_BASE_URL=http://localhost:5678
N8N_API_KEY=your-n8n-api-key

# Cost Management
MAX_COST_PER_VIDEO=3.00
WARNING_COST_THRESHOLD=2.50
DAILY_COST_BUDGET=150.00
```

---

## 4. Project Structure

### Complete Directory Layout

```
ytempire-backend/
├── app/
│   ├── __init__.py
│   ├── main.py                    # FastAPI application entry point
│   ├── config.py                  # Configuration management
│   ├── database.py                # Database connection and session
│   ├── dependencies.py            # Shared dependencies
│   │
│   ├── api/
│   │   └── v1/
│   │       ├── __init__.py
│   │       ├── router.py          # Main API router
│   │       └── endpoints/
│   │           ├── __init__.py
│   │           ├── auth.py        # Authentication endpoints
│   │           ├── channels.py    # Channel management
│   │           ├── videos.py      # Video operations
│   │           ├── analytics.py   # Analytics endpoints
│   │           ├── costs.py       # Cost tracking
│   │           ├── webhooks.py    # N8N webhooks
│   │           ├── youtube.py     # YouTube OAuth
│   │           └── health.py      # Health checks
│   │
│   ├── core/
│   │   ├── __init__.py
│   │   ├── security.py            # JWT and password handling
│   │   ├── exceptions.py          # Custom exceptions
│   │   ├── middleware.py          # Custom middleware
│   │   ├── rate_limit.py          # Rate limiting logic
│   │   └── cache.py               # Redis cache wrapper
│   │
│   ├── models/
│   │   ├── __init__.py
│   │   ├── base.py                # Base SQLAlchemy model
│   │   ├── user.py                # User model
│   │   ├── channel.py             # Channel model
│   │   ├── video.py               # Video model
│   │   ├── youtube_account.py     # YouTube accounts
│   │   └── cost.py                # Cost tracking models
│   │
│   ├── schemas/
│   │   ├── __init__.py
│   │   ├── user.py                # User Pydantic schemas
│   │   ├── channel.py             # Channel schemas
│   │   ├── video.py               # Video schemas
│   │   ├── cost.py                # Cost schemas
│   │   └── common.py              # Common response schemas
│   │
│   ├── services/
│   │   ├── __init__.py
│   │   ├── youtube_service.py     # YouTube API operations
│   │   ├── cost_service.py        # Cost tracking service
│   │   ├── auth_service.py        # Authentication service
│   │   └── n8n_service.py         # N8N integration
│   │
│   ├── tasks/
│   │   ├── __init__.py
│   │   ├── celery_app.py          # Celery configuration
│   │   └── video_tasks.py         # Async video processing
│   │
│   └── utils/
│       ├── __init__.py
│       ├── logger.py              # Logging configuration
│       ├── validators.py          # Custom validators
│       └── helpers.py             # Helper functions
│
├── alembic/
│   ├── alembic.ini                # Alembic configuration
│   ├── env.py                     # Migration environment
│   └── versions/                  # Migration files
│
├── tests/
│   ├── __init__.py
│   ├── conftest.py                # Pytest configuration
│   ├── test_auth.py               # Auth tests
│   ├── test_channels.py           # Channel tests
│   ├── test_videos.py             # Video tests
│   └── test_youtube.py            # YouTube integration tests
│
├── scripts/
│   ├── init_db.py                 # Database initialization
│   ├── seed_data.py               # Seed test data
│   └── test_youtube_auth.py       # Test YouTube OAuth
│
├── docker/
│   ├── Dockerfile                 # Application container
│   └── docker-compose.yml         # Full stack setup
│
├── .env.example                   # Environment template
├── .gitignore                     # Git ignore file
├── requirements.txt               # Production dependencies
├── requirements-dev.txt           # Development dependencies
├── pyproject.toml                 # Python project configuration
└── README.md                      # Project documentation
```

---

## 5. Core Technologies & Dependencies

### requirements.txt

```txt
# Core Framework
fastapi==0.109.0
uvicorn[standard]==0.27.0
python-multipart==0.0.6

# Database
sqlalchemy==2.0.25
alembic==1.13.1
asyncpg==0.29.0
psycopg2-binary==2.9.9

# Redis
redis==5.0.1
hiredis==2.3.2

# Authentication & Security
python-jose[cryptography]==3.3.0
passlib[bcrypt]==1.7.4
python-decouple==3.8
cryptography==41.0.7

# Validation
pydantic==2.5.3
pydantic-settings==2.1.0
email-validator==2.1.0

# YouTube API
google-api-python-client==2.111.0
google-auth-httplib2==0.2.0
google-auth-oauthlib==1.2.0

# OpenAI
openai==1.6.1

# Stripe
stripe==7.10.0

# HTTP Clients
httpx==0.26.0
aiohttp==3.9.1

# Task Queue
celery==5.3.4
kombu==5.3.4

# Utilities
python-dateutil==2.8.2
pytz==2023.3
uuid==1.30

# Monitoring
prometheus-client==0.19.0
opentelemetry-api==1.22.0
opentelemetry-sdk==1.22.0

# Logging
structlog==24.1.0
colorama==0.4.6
```

### requirements-dev.txt

```txt
# Testing
pytest==7.4.4
pytest-asyncio==0.23.3
pytest-cov==4.1.0
pytest-mock==3.12.0
pytest-env==1.1.3
httpx==0.26.0  # For test client
faker==22.0.0

# Code Quality
black==23.12.1
flake8==7.0.0
mypy==1.8.0
isort==5.13.2
pylint==3.0.3

# Development Tools
ipython==8.19.0
ipdb==0.13.13
watchdog==3.0.0

# Documentation
mkdocs==1.5.3
mkdocs-material==9.5.3
```

---

## 6. Database Design & Models

### 6.1 Database Schema Overview

```sql
-- Enable UUID extension
CREATE EXTENSION IF NOT EXISTS "uuid-ossp";

-- Users table
CREATE TABLE users (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    email VARCHAR(255) UNIQUE NOT NULL,
    password_hash VARCHAR(255) NOT NULL,
    full_name VARCHAR(255),
    role VARCHAR(20) DEFAULT 'user',
    tier VARCHAR(20) DEFAULT 'free',
    is_active BOOLEAN DEFAULT true,
    is_verified BOOLEAN DEFAULT false,
    channel_limit INTEGER DEFAULT 5,
    daily_video_limit INTEGER DEFAULT 10,
    stripe_customer_id VARCHAR(255) UNIQUE,
    stripe_subscription_id VARCHAR(255),
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

CREATE INDEX idx_users_email ON users(email);
CREATE INDEX idx_users_stripe_customer ON users(stripe_customer_id);

-- YouTube accounts table (15 accounts total)
CREATE TABLE youtube_accounts (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    email VARCHAR(255) UNIQUE NOT NULL,
    channel_id VARCHAR(255) UNIQUE,
    channel_handle VARCHAR(255),
    is_active BOOLEAN DEFAULT true,
    is_reserve BOOLEAN DEFAULT false,
    health_score DECIMAL(5,2) DEFAULT 100.00,
    credentials JSONB NOT NULL,
    token_expires_at TIMESTAMP WITH TIME ZONE,
    quota_used_today INTEGER DEFAULT 0,
    uploads_today INTEGER DEFAULT 0,
    last_reset_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    total_uploads INTEGER DEFAULT 0,
    total_errors INTEGER DEFAULT 0,
    consecutive_errors INTEGER DEFAULT 0,
    last_upload_at TIMESTAMP WITH TIME ZONE,
    last_error_at TIMESTAMP WITH TIME ZONE,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

CREATE INDEX idx_youtube_accounts_active ON youtube_accounts(is_active, is_reserve);
CREATE INDEX idx_youtube_accounts_health ON youtube_accounts(health_score DESC);

-- Channels table
CREATE TABLE channels (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    user_id UUID NOT NULL REFERENCES users(id) ON DELETE CASCADE,
    youtube_account_id UUID REFERENCES youtube_accounts(id),
    youtube_channel_id VARCHAR(255) UNIQUE,
    youtube_channel_handle VARCHAR(255),
    name VARCHAR(255) NOT NULL,
    description TEXT,
    niche VARCHAR(100) NOT NULL,
    status VARCHAR(20) DEFAULT 'pending_oauth',
    automation_enabled BOOLEAN DEFAULT true,
    daily_video_limit INTEGER DEFAULT 5,
    upload_time_utc VARCHAR(5),
    settings JSONB DEFAULT '{}',
    metadata JSONB DEFAULT '{}',
    total_videos INTEGER DEFAULT 0,
    total_views BIGINT DEFAULT 0,
    total_subscribers INTEGER DEFAULT 0,
    total_revenue_cents BIGINT DEFAULT 0,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

CREATE INDEX idx_channels_user ON channels(user_id);
CREATE INDEX idx_channels_status ON channels(status);
CREATE INDEX idx_channels_niche ON channels(niche);

-- Videos table
CREATE TABLE videos (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    channel_id UUID NOT NULL REFERENCES channels(id) ON DELETE CASCADE,
    title VARCHAR(500) NOT NULL,
    description TEXT,
    script TEXT,
    tags JSONB DEFAULT '[]',
    category_id VARCHAR(10),
    youtube_video_id VARCHAR(255) UNIQUE,
    youtube_url VARCHAR(500),
    youtube_account_used UUID REFERENCES youtube_accounts(id),
    status VARCHAR(20) DEFAULT 'queued',
    complexity VARCHAR(20) DEFAULT 'simple',
    priority INTEGER DEFAULT 5,
    generation_time_seconds INTEGER,
    quality_score DECIMAL(3,2),
    optimization_level VARCHAR(20) DEFAULT 'standard',
    total_cost DECIMAL(10,2) DEFAULT 0.00,
    cost_breakdown JSONB DEFAULT '{}',
    video_path VARCHAR(500),
    thumbnail_path VARCHAR(500),
    audio_path VARCHAR(500),
    metadata JSONB DEFAULT '{}',
    error_message TEXT,
    queued_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    processing_started_at TIMESTAMP WITH TIME ZONE,
    processing_completed_at TIMESTAMP WITH TIME ZONE,
    published_at TIMESTAMP WITH TIME ZONE,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

CREATE INDEX idx_videos_channel ON videos(channel_id);
CREATE INDEX idx_videos_status ON videos(status);
CREATE INDEX idx_videos_youtube_id ON videos(youtube_video_id);
CREATE INDEX idx_videos_created ON videos(created_at DESC);

-- Cost tracking table
CREATE TABLE video_costs (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    video_id UUID NOT NULL REFERENCES videos(id) ON DELETE CASCADE,
    service VARCHAR(50) NOT NULL,
    amount DECIMAL(10,4) NOT NULL,
    timestamp TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    metadata JSONB DEFAULT '{}'
);

CREATE INDEX idx_video_costs_video ON video_costs(video_id);
CREATE INDEX idx_video_costs_service ON video_costs(service);
CREATE INDEX idx_video_costs_timestamp ON video_costs(timestamp DESC);
```

### 6.2 SQLAlchemy Models

```python
# app/models/base.py
from sqlalchemy import Column, DateTime, String
from sqlalchemy.sql import func
from sqlalchemy.ext.declarative import declared_attr
import uuid

class BaseModel:
    """Base model with common fields"""
    
    @declared_attr
    def __tablename__(cls):
        """Generate table name from class name"""
        name = cls.__name__
        return ''.join(['_' + c.lower() if c.isupper() else c for c in name]).lstrip('_') + 's'
    
    id = Column(String(36), primary_key=True, default=lambda: str(uuid.uuid4()))
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    updated_at = Column(DateTime(timezone=True), server_default=func.now(), onupdate=func.now())
```

```python
# app/models/user.py
from sqlalchemy import Column, String, Boolean, Integer, Enum as SQLEnum
from sqlalchemy.orm import relationship
import enum

from app.database import Base
from app.models.base import BaseModel

class UserRole(str, enum.Enum):
    USER = "user"
    ADMIN = "admin"

class UserTier(str, enum.Enum):
    FREE = "free"
    STARTER = "starter"
    GROWTH = "growth"
    SCALE = "scale"

class User(Base, BaseModel):
    __tablename__ = "users"
    
    email = Column(String(255), unique=True, nullable=False, index=True)
    password_hash = Column(String(255), nullable=False)
    full_name = Column(String(255))
    role = Column(SQLEnum(UserRole), default=UserRole.USER)
    tier = Column(SQLEnum(UserTier), default=UserTier.FREE)
    is_active = Column(Boolean, default=True)
    is_verified = Column(Boolean, default=False)
    channel_limit = Column(Integer, default=5)
    daily_video_limit = Column(Integer, default=10)
    stripe_customer_id = Column(String(255), unique=True)
    stripe_subscription_id = Column(String(255))
    
    # Relationships
    channels = relationship("Channel", back_populates="user", cascade="all, delete-orphan")
```

---

## 7. Authentication & Security

### 7.1 JWT Implementation

```python
# app/core/security.py
from datetime import datetime, timedelta
from typing import Optional, Dict, Any
from jose import JWTError, jwt
from passlib.context import CryptContext
import secrets

from app.config import settings

pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto", bcrypt__rounds=12)

class SecurityService:
    """Handles authentication and authorization"""
    
    @staticmethod
    def verify_password(plain_password: str, hashed_password: str) -> bool:
        return pwd_context.verify(plain_password, hashed_password)
    
    @staticmethod
    def get_password_hash(password: str) -> str:
        return pwd_context.hash(password)
    
    @staticmethod
    def create_access_token(data: Dict[str, Any], expires_delta: Optional[timedelta] = None) -> str:
        to_encode = data.copy()
        expire = datetime.utcnow() + (expires_delta or timedelta(minutes=settings.ACCESS_TOKEN_EXPIRE_MINUTES))
        
        to_encode.update({
            "exp": expire,
            "type": "access",
            "iat": datetime.utcnow(),
            "jti": secrets.token_urlsafe(16)
        })
        
        return jwt.encode(to_encode, settings.SECRET_KEY, algorithm=settings.ALGORITHM)
    
    @staticmethod
    def decode_token(token: str) -> Dict[str, Any]:
        try:
            payload = jwt.decode(token, settings.SECRET_KEY, algorithms=[settings.ALGORITHM])
            return payload
        except JWTError as e:
            raise ValueError(f"Invalid token: {e}")
```

### 7.2 Authentication Middleware

```python
# app/core/middleware.py
from fastapi import Request, HTTPException, status
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
import time
import uuid
import logging

logger = logging.getLogger(__name__)

class AuthMiddleware:
    """Authentication middleware for protected routes"""
    
    async def __call__(self, request: Request, call_next):
        # Skip auth for public endpoints
        public_paths = ["/", "/health", "/docs", "/redoc", "/openapi.json"]
        if request.url.path in public_paths or request.url.path.startswith("/api/v1/auth/"):
            return await call_next(request)
        
        # Verify authorization header
        auth_header = request.headers.get("Authorization")
        if not auth_header or not auth_header.startswith("Bearer "):
            return JSONResponse(
                status_code=status.HTTP_401_UNAUTHORIZED,
                content={"detail": "Missing authentication token"}
            )
        
        return await call_next(request)

class RequestIDMiddleware:
    """Add unique request ID for tracing"""
    
    async def __call__(self, request: Request, call_next):
        request_id = str(uuid.uuid4())
        request.state.request_id = request_id
        
        response = await call_next(request)
        response.headers["X-Request-ID"] = request_id
        
        return response

class MetricsMiddleware:
    """Track request metrics"""
    
    async def __call__(self, request: Request, call_next):
        start_time = time.time()
        
        response = await call_next(request)
        
        process_time = time.time() - start_time
        response.headers["X-Process-Time"] = str(process_time)
        
        # Log slow requests
        if process_time > 1.0:
            logger.warning(f"Slow request: {request.url.path} took {process_time:.2f}s")
        
        return response
```

---

## 8. API Endpoints Implementation

### 8.1 Main Application

```python
# app/main.py
from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import logging

from app.api.v1.router import api_router
from app.config import settings
from app.database import init_db
from app.core.middleware import AuthMiddleware, RequestIDMiddleware, MetricsMiddleware

logger = logging.getLogger(__name__)

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifecycle management"""
    logger.info("Starting YTEMPIRE API Server...")
    await init_db()
    yield
    logger.info("Shutting down YTEMPIRE API Server...")

app = FastAPI(
    title="YTEMPIRE API",
    version="1.0.0",
    openapi_url=f"{settings.API_V1_STR}/openapi.json",
    lifespan=lifespan
)

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Add custom middleware
app.add_middleware(RequestIDMiddleware)
app.add_middleware(MetricsMiddleware)
app.add_middleware(AuthMiddleware)

# Include routers
app.include_router(api_router, prefix=settings.API_V1_STR)

@app.get("/")
async def root():
    return {"service": "YTEMPIRE API", "version": "1.0.0", "status": "operational"}

@app.get("/health")
async def health():
    return {"status": "healthy"}
```

### 8.2 Channel Management Endpoints

```python
# app/api/v1/endpoints/channels.py
from typing import List, Optional
from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, func

from app.database import get_db
from app.core.dependencies import get_current_user
from app.models.user import User
from app.models.channel import Channel
from app.schemas.channel import ChannelCreate, ChannelResponse, ChannelUpdate

router = APIRouter(prefix="/channels", tags=["Channels"])

@router.get("/", response_model=List[ChannelResponse])
async def list_channels(
    page: int = Query(1, ge=1),
    limit: int = Query(10, ge=1, le=100),
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db)
):
    """List all channels for current user"""
    query = select(Channel).where(Channel.user_id == current_user.id)
    query = query.offset((page - 1) * limit).limit(limit)
    
    result = await db.execute(query)
    channels = result.scalars().all()
    
    return channels

@router.post("/", response_model=ChannelResponse, status_code=201)
async def create_channel(
    channel_data: ChannelCreate,
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db)
):
    """Create new channel"""
    # Check channel limit
    count = await db.scalar(
        select(func.count()).select_from(Channel).where(Channel.user_id == current_user.id)
    )
    
    if count >= current_user.channel_limit:
        raise HTTPException(403, f"Channel limit reached ({current_user.channel_limit})")
    
    channel = Channel(
        user_id=current_user.id,
        name=channel_data.name,
        niche=channel_data.niche,
        description=channel_data.description
    )
    
    db.add(channel)
    await db.commit()
    await db.refresh(channel)
    
    return channel

@router.get("/{channel_id}", response_model=ChannelResponse)
async def get_channel(
    channel_id: str,
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db)
):
    """Get channel details"""
    channel = await db.get(Channel, channel_id)
    
    if not channel or channel.user_id != current_user.id:
        raise HTTPException(404, "Channel not found")
    
    return channel

@router.patch("/{channel_id}", response_model=ChannelResponse)
async def update_channel(
    channel_id: str,
    update_data: ChannelUpdate,
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db)
):
    """Update channel settings"""
    channel = await db.get(Channel, channel_id)
    
    if not channel or channel.user_id != current_user.id:
        raise HTTPException(404, "Channel not found")
    
    for field, value in update_data.dict(exclude_unset=True).items():
        setattr(channel, field, value)
    
    await db.commit()
    await db.refresh(channel)
    
    return channel
```

### 8.3 Video Management Endpoints

```python
# app/api/v1/endpoints/videos.py
from typing import List
from fastapi import APIRouter, Depends, HTTPException, BackgroundTasks
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select

from app.database import get_db
from app.core.dependencies import get_current_user
from app.models.video import Video
from app.schemas.video import VideoCreate, VideoResponse
from app.services.cost_service import CostService
from app.tasks.video_tasks import process_video_task

router = APIRouter(prefix="/videos", tags=["Videos"])

@router.post("/generate", response_model=VideoResponse, status_code=202)
async def generate_video(
    video_data: VideoCreate,
    background_tasks: BackgroundTasks,
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db)
):
    """Queue video for generation"""
    # Verify channel ownership
    channel = await db.get(Channel, video_data.channel_id)
    if not channel or channel.user_id != current_user.id:
        raise HTTPException(404, "Channel not found")
    
    # Check daily limit
    today_count = await db.scalar(
        select(func.count()).select_from(Video)
        .where(
            Video.channel_id == video_data.channel_id,
            Video.created_at >= datetime.utcnow().date()
        )
    )
    
    if today_count >= channel.daily_video_limit:
        raise HTTPException(403, "Daily video limit reached")
    
    # Estimate cost
    cost_service = CostService()
    estimated_cost = await cost_service.estimate_video_cost(video_data.optimization_level)
    
    if estimated_cost["total"] > settings.MAX_COST_PER_VIDEO:
        raise HTTPException(400, f"Estimated cost ${estimated_cost['total']:.2f} exceeds limit")
    
    # Create video record
    video = Video(
        channel_id=video_data.channel_id,
        title=video_data.title,
        description=video_data.description,
        tags=video_data.tags,
        optimization_level=video_data.optimization_level,
        priority=video_data.priority or 5,
        status="queued"
    )
    
    db.add(video)
    await db.commit()
    await db.refresh(video)
    
    # Queue for processing
    background_tasks.add_task(process_video_task, str(video.id))
    
    return video

@router.get("/{video_id}", response_model=VideoResponse)
async def get_video(
    video_id: str,
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db)
):
    """Get video details"""
    video = await db.get(Video, video_id)
    
    if not video:
        raise HTTPException(404, "Video not found")
    
    # Verify ownership
    channel = await db.get(Channel, video.channel_id)
    if channel.user_id != current_user.id:
        raise HTTPException(403, "Access denied")
    
    return video
```

---

## 9. YouTube Multi-Account Integration

### 9.1 Account Management Strategy

```python
# app/services/youtube_service.py
from typing import List, Optional, Dict, Any
from datetime import datetime, timedelta
import asyncio
from google.oauth2.credentials import Credentials
from googleapiclient.discovery import build
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, and_

from app.models.youtube_account import YouTubeAccount
from app.config import settings

class YouTubeAccountManager:
    """
    Manages 15 YouTube accounts (12 active + 3 reserve)
    Implements intelligent rotation and health monitoring
    """
    
    def __init__(self):
        self.total_accounts = 15
        self.active_accounts = 12
        self.reserve_accounts = 3
        self.daily_upload_limit = 5
        self.quota_per_account = 10000
        
    async def get_best_account(self, db: AsyncSession) -> Optional[YouTubeAccount]:
        """
        Select optimal account for upload based on:
        1. Daily upload count < 5
        2. Quota usage < 80%
        3. Health score
        4. No recent errors
        """
        # Try active accounts first
        query = select(YouTubeAccount).where(
            and_(
                YouTubeAccount.is_active == True,
                YouTubeAccount.is_reserve == False,
                YouTubeAccount.uploads_today < self.daily_upload_limit,
                YouTubeAccount.quota_used_today < (self.quota_per_account * 0.8),
                YouTubeAccount.consecutive_errors < 3
            )
        ).order_by(
            YouTubeAccount.uploads_today.asc(),
            YouTubeAccount.health_score.desc()
        )
        
        result = await db.execute(query)
        account = result.scalar_one_or_none()
        
        if not account:
            # Fallback to reserve accounts
            reserve_query = select(YouTubeAccount).where(
                and_(
                    YouTubeAccount.is_reserve == True,
                    YouTubeAccount.is_active == True,
                    YouTubeAccount.uploads_today < 2
                )
            ).order_by(YouTubeAccount.health_score.desc())
            
            result = await db.execute(reserve_query)
            account = result.scalar_one_or_none()
        
        return account
    
    async def update_account_health(
        self, 
        account_id: str, 
        success: bool,
        db: AsyncSession
    ):
        """Update account health score based on operation result"""
        account = await db.get(YouTubeAccount, account_id)
        
        if success:
            # Improve health score
            account.health_score = min(100, account.health_score + 1)
            account.consecutive_errors = 0
        else:
            # Decrease health score
            account.health_score = max(0, account.health_score - 10)
            account.consecutive_errors += 1
            account.total_errors += 1
            account.last_error_at = datetime.utcnow()
            
            # Deactivate if too many errors
            if account.consecutive_errors >= 5:
                account.is_active = False
        
        await db.commit()
    
    async def reset_daily_quotas(self, db: AsyncSession):
        """Reset daily quotas for all accounts (run at midnight PST)"""
        await db.execute(
            update(YouTubeAccount).values(
                quota_used_today=0,
                uploads_today=0,
                last_reset_at=datetime.utcnow()
            )
        )
        await db.commit()
```

### 9.2 OAuth Implementation

```python
# app/api/v1/endpoints/youtube.py
from fastapi import APIRouter, Depends, Request
from fastapi.responses import RedirectResponse
from google_auth_oauthlib.flow import Flow
from sqlalchemy.ext.asyncio import AsyncSession

from app.database import get_db
from app.config import settings
from app.models.youtube_account import YouTubeAccount

router = APIRouter(prefix="/youtube", tags=["YouTube"])

@router.get("/oauth/authorize")
async def youtube_oauth_authorize(
    email: str,
    is_reserve: bool = False,
    db: AsyncSession = Depends(get_db)
):
    """Initiate YouTube OAuth flow"""
    flow = Flow.from_client_config(
        {
            "web": {
                "client_id": settings.YOUTUBE_CLIENT_ID,
                "client_secret": settings.YOUTUBE_CLIENT_SECRET,
                "auth_uri": "https://accounts.google.com/o/oauth2/auth",
                "token_uri": "https://oauth2.googleapis.com/token",
                "redirect_uris": [settings.YOUTUBE_REDIRECT_URI]
            }
        },
        scopes=settings.YOUTUBE_SCOPES
    )
    
    flow.redirect_uri = settings.YOUTUBE_REDIRECT_URI
    
    authorization_url, state = flow.authorization_url(
        access_type='offline',
        include_granted_scopes='true',
        prompt='consent',
        login_hint=email
    )
    
    # Store state in Redis for callback
    await redis_client.set(f"oauth_state:{state}", email, ex=600)
    await redis_client.set(f"oauth_reserve:{state}", str(is_reserve), ex=600)
    
    return RedirectResponse(authorization_url)

@router.get("/oauth/callback")
async def youtube_oauth_callback(
    code: str,
    state: str,
    db: AsyncSession = Depends(get_db)
):
    """Handle YouTube OAuth callback"""
    # Retrieve email from state
    email = await redis_client.get(f"oauth_state:{state}")
    is_reserve = await redis_client.get(f"oauth_reserve:{state}") == "True"
    
    if not email:
        raise HTTPException(400, "Invalid OAuth state")
    
    # Exchange code for tokens
    flow = Flow.from_client_config(
        {
            "web": {
                "client_id": settings.YOUTUBE_CLIENT_ID,
                "client_secret": settings.YOUTUBE_CLIENT_SECRET,
                "auth_uri": "https://accounts.google.com/o/oauth2/auth",
                "token_uri": "https://oauth2.googleapis.com/token",
                "redirect_uris": [settings.YOUTUBE_REDIRECT_URI]
            }
        },
        scopes=settings.YOUTUBE_SCOPES,
        state=state
    )
    
    flow.redirect_uri = settings.YOUTUBE_REDIRECT_URI
    flow.fetch_token(code=code)
    
    credentials = flow.credentials
    
    # Get channel info
    youtube = build('youtube', 'v3', credentials=credentials)
    channels_response = youtube.channels().list(
        part="snippet",
        mine=True
    ).execute()
    
    channel_info = channels_response['items'][0] if channels_response['items'] else None
    
    # Store account
    account = YouTubeAccount(
        email=email,
        channel_id=channel_info['id'] if channel_info else None,
        channel_handle=channel_info['snippet']['customUrl'] if channel_info else None,
        is_reserve=is_reserve,
        credentials={
            'token': credentials.token,
            'refresh_token': credentials.refresh_token,
            'token_uri': credentials.token_uri,
            'client_id': credentials.client_id,
            'client_secret': credentials.client_secret,
            'scopes': credentials.scopes
        },
        token_expires_at=credentials.expiry
    )
    
    db.add(account)
    await db.commit()
    
    return {"message": "YouTube account connected successfully", "email": email}
```

---

## 10. Cost Tracking System

### 10.1 Real-time Cost Monitoring

```python
# app/services/cost_service.py
from typing import Dict, Any
from datetime import datetime
import redis.asyncio as redis
from sqlalchemy.ext.asyncio import AsyncSession

from app.config import settings
from app.models.video import Video
from app.models.cost import VideoCost

class CostService:
    """
    Real-time cost tracking with hard $3/video limit
    """
    
    COST_CONFIG = {
        "openai": {
            "gpt-3.5-turbo": 0.002,  # per 1k tokens
            "gpt-4": 0.03             # per 1k tokens
        },
        "tts": {
            "google": 0.016,          # per 1k chars
            "elevenlabs": 0.30        # per 1k chars
        }
    }
    
    def __init__(self):
        self.redis_client = None
        
    async def initialize(self):
        self.redis_client = await redis.from_url(settings.REDIS_URL)
    
    async def track_cost(
        self,
        video_id: str,
        service: str,
        amount: float,
        db: AsyncSession
    ) -> Dict[str, Any]:
        """Track cost and enforce limits"""
        # Update database
        cost_record = VideoCost(
            video_id=video_id,
            service=service,
            amount=amount
        )
        db.add(cost_record)
        
        # Update video total
        video = await db.get(Video, video_id)
        video.total_cost = (video.total_cost or 0) + amount
        
        # Track in Redis
        video_key = f"cost:video:{video_id}"
        await self.redis_client.hincrbyfloat(video_key, "total", amount)
        await self.redis_client.hincrbyfloat(video_key, service, amount)
        await self.redis_client.expire(video_key, 86400)
        
        total = float(await self.redis_client.hget(video_key, "total") or 0)
        
        # Check limits
        status = "ok"
        if total >= settings.MAX_COST_PER_VIDEO:
            status = "exceeded"
            # STOP PROCESSING
            raise Exception(f"Cost limit exceeded: ${total:.2f}")
        elif total >= settings.WARNING_COST_THRESHOLD:
            status = "warning"
            # Switch to economy mode
        
        await db.commit()
        
        return {
            "video_id": video_id,
            "service": service,
            "amount": amount,
            "total": total,
            "status": status,
            "remaining_budget": max(0, settings.MAX_COST_PER_VIDEO - total)
        }
    
    async def get_daily_summary(self) -> Dict[str, Any]:
        """Get daily cost summary"""
        date_key = f"cost:daily:{datetime.utcnow().date()}"
        costs = await self.redis_client.hgetall(date_key)
        
        total = sum(float(v) for v in costs.values())
        
        return {
            "date": str(datetime.utcnow().date()),
            "total_cost": total,
            "budget_remaining": settings.DAILY_COST_BUDGET - total,
            "by_service": costs,
            "videos_remaining": int((settings.DAILY_COST_BUDGET - total) / 2.0)
        }
```

---

## 11. N8N Webhook Integration

### 11.1 Webhook Endpoints

```python
# app/api/v1/endpoints/webhooks.py
from fastapi import APIRouter, Request, HTTPException
from sqlalchemy.ext.asyncio import AsyncSession
import hmac
import hashlib

from app.database import get_db
from app.config import settings

router = APIRouter(prefix="/webhooks", tags=["Webhooks"])

@router.post("/n8n/video-complete")
async def n8n_video_complete(
    request: Request,
    db: AsyncSession = Depends(get_db)
):
    """Handle N8N workflow completion"""
    # Verify webhook signature
    signature = request.headers.get("X-N8N-Signature")
    body = await request.body()
    
    expected_sig = hmac.new(
        settings.N8N_API_KEY.encode(),
        body,
        hashlib.sha256
    ).hexdigest()
    
    if signature != expected_sig:
        raise HTTPException(403, "Invalid signature")
    
    data = await request.json()
    video_id = data.get("video_id")
    status = data.get("status")
    
    # Update video status
    video = await db.get(Video, video_id)
    if video:
        video.status = status
        video.processing_completed_at = datetime.utcnow()
        await db.commit()
    
    return {"status": "received"}

@router.post("/n8n/cost-alert")
async def n8n_cost_alert(
    request: Request,
    db: AsyncSession = Depends(get_db)
):
    """Handle cost threshold alerts from N8N"""
    data = await request.json()
    
    video_id = data.get("video_id")
    current_cost = data.get("current_cost")
    
    if current_cost >= settings.MAX_COST_PER_VIDEO:
        # Emergency stop
        video = await db.get(Video, video_id)
        if video:
            video.status = "cancelled"
            video.error_message = f"Cost limit exceeded: ${current_cost:.2f}"
            await db.commit()
        
        # Notify N8N to stop processing
        return {"action": "STOP", "reason": "cost_exceeded"}
    
    return {"action": "CONTINUE"}
```

---

## 12. Performance Optimization

### 12.1 Database Query Optimization

```python
# app/utils/performance.py
from sqlalchemy import select
from sqlalchemy.orm import selectinload, joinedload

class QueryOptimizer:
    """Database query optimization patterns"""
    
    @staticmethod
    def eager_load_relationships():
        """Use eager loading to prevent N+1 queries"""
        # Bad: N+1 query problem
        # channels = await db.execute(select(Channel))
        # for channel in channels:
        #     videos = await db.execute(select(Video).where(Video.channel_id == channel.id))
        
        # Good: Single query with join
        query = select(Channel).options(
            selectinload(Channel.videos),
            selectinload(Channel.user)
        )
        return query
    
    @staticmethod
    def use_indexes():
        """Ensure queries use indexes"""
        # Always filter on indexed columns
        query = select(Video).where(
            Video.status == "queued",  # Indexed
            Video.channel_id == channel_id  # Indexed
        ).order_by(
            Video.priority.desc(),  # Consider index
            Video.created_at.asc()  # Indexed
        )
        return query
    
    @staticmethod
    def limit_columns():
        """Select only needed columns"""
        # Bad: SELECT * 
        # query = select(Video)
        
        # Good: Select specific columns
        query = select(
            Video.id,
            Video.title,
            Video.status,
            Video.total_cost
        )
        return query
```

### 12.2 Caching Strategy

```python
# app/core/cache.py
import json
from typing import Optional, Any
import redis.asyncio as redis

class CacheService:
    """Redis caching service"""
    
    def __init__(self):
        self.redis = None
        self.default_ttl = 300  # 5 minutes
    
    async def initialize(self):
        self.redis = await redis.from_url(settings.REDIS_URL)
    
    async def get(self, key: str) -> Optional[Any]:
        """Get value from cache"""
        value = await self.redis.get(key)
        if value:
            return json.loads(value)
        return None
    
    async def set(self, key: str, value: Any, ttl: int = None):
        """Set value in cache"""
        await self.redis.set(
            key,
            json.dumps(value),
            ex=ttl or self.default_ttl
        )
    
    async def invalidate_pattern(self, pattern: str):
        """Invalidate all keys matching pattern"""
        cursor = 0
        while True:
            cursor, keys = await self.redis.scan(cursor, match=pattern)
            if keys:
                await self.redis.delete(*keys)
            if cursor == 0:
                break
    
    # Cache decorators
    def cache_result(ttl: int = 300):
        """Decorator to cache function results"""
        def decorator(func):
            async def wrapper(*args, **kwargs):
                # Generate cache key
                key = f"{func.__name__}:{str(args)}:{str(kwargs)}"
                
                # Try cache
                cached = await cache_service.get(key)
                if cached:
                    return cached
                
                # Execute function
                result = await func(*args, **kwargs)
                
                # Store in cache
                await cache_service.set(key, result, ttl)
                
                return result
            return wrapper
        return decorator
```

---

## 13. Testing Strategy

### 13.1 Test Configuration

```python
# tests/conftest.py
import pytest
import asyncio
from typing import AsyncGenerator
from fastapi.testclient import TestClient
from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession
from sqlalchemy.orm import sessionmaker

from app.main import app
from app.database import Base, get_db
from app.config import settings

# Test database
TEST_DATABASE_URL = "postgresql+asyncpg://test_user:test_pass@localhost/test_db"

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
    
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.drop_all)
    
    await engine.dispose()

@pytest.fixture
async def test_db(test_engine) -> AsyncGenerator[AsyncSession, None]:
    """Create test database session"""
    TestSessionLocal = sessionmaker(
        test_engine,
        class_=AsyncSession,
        expire_on_commit=False
    )
    
    async with TestSessionLocal() as session:
        yield session

@pytest.fixture
def test_client(test_db):
    """Create test client"""
    def override_get_db():
        yield test_db
    
    app.dependency_overrides[get_db] = override_get_db
    
    with TestClient(app) as client:
        yield client
```

### 13.2 API Tests

```python
# tests/test_channels.py
import pytest
from httpx import AsyncClient

@pytest.mark.asyncio
async def test_create_channel(test_client, test_user_token):
    """Test channel creation"""
    headers = {"Authorization": f"Bearer {test_user_token}"}
    
    response = await test_client.post(
        "/api/v1/channels",
        json={
            "name": "Test Channel",
            "niche": "Technology",
            "description": "Test description"
        },
        headers=headers
    )
    
    assert response.status_code == 201
    data = response.json()
    assert data["name"] == "Test Channel"
    assert data["niche"] == "Technology"
    assert data["status"] == "pending_oauth"

@pytest.mark.asyncio
async def test_list_channels(test_client, test_user_token):
    """Test listing channels"""
    headers = {"Authorization": f"Bearer {test_user_token}"}
    
    response = await test_client.get(
        "/api/v1/channels",
        headers=headers
    )
    
    assert response.status_code == 200
    data = response.json()
    assert isinstance(data, list)

@pytest.mark.asyncio
async def test_channel_limit_enforcement(test_client, test_user_token):
    """Test channel limit is enforced"""
    headers = {"Authorization": f"Bearer {test_user_token}"}
    
    # Create channels up to limit
    for i in range(5):
        response = await test_client.post(
            "/api/v1/channels",
            json={"name": f"Channel {i}", "niche": "Tech"},
            headers=headers
        )
        assert response.status_code == 201
    
    # Try to exceed limit
    response = await test_client.post(
        "/api/v1/channels",
        json={"name": "Excess Channel", "niche": "Tech"},
        headers=headers
    )
    
    assert response.status_code == 403
    assert "limit reached" in response.json()["detail"].lower()
```

---

## 14. Deployment Guide

### 14.1 Docker Configuration

```dockerfile
# Dockerfile
FROM python:3.11-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    gcc \
    postgresql-client \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application
COPY . .

# Run migrations and start server
CMD ["sh", "-c", "alembic upgrade head && uvicorn app.main:app --host 0.0.0.0 --port 8000"]
```

### 14.2 Docker Compose

```yaml
# docker-compose.yml
version: '3.8'

services:
  db:
    image: postgres:14
    environment:
      POSTGRES_USER: ytempire_user
      POSTGRES_PASSWORD: ytempire_pass
      POSTGRES_DB: ytempire_db
    volumes:
      - postgres_data:/var/lib/postgresql/data
    ports:
      - "5432:5432"
  
  redis:
    image: redis:7-alpine
    ports:
      - "6379:6379"
    volumes:
      - redis_data:/data
  
  app:
    build: .
    ports:
      - "8000:8000"
    environment:
      DATABASE_URL: postgresql://ytempire_user:ytempire_pass@db:5432/ytempire_db
      REDIS_URL: redis://redis:6379/0
    depends_on:
      - db
      - redis
    volumes:
      - ./app:/app/app
    command: uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload

volumes:
  postgres_data:
  redis_data:
```

### 14.3 Production Deployment

```bash
# Production deployment script
#!/bin/bash

# 1. Set environment
export ENVIRONMENT=production

# 2. Run database migrations
alembic upgrade head

# 3. Start application with Gunicorn
gunicorn app.main:app \
  --workers 4 \
  --worker-class uvicorn.workers.UvicornWorker \
  --bind 0.0.0.0:8000 \
  --access-logfile - \
  --error-logfile - \
  --log-level info
```

---

## 15. Monitoring & Logging

### 15.1 Structured Logging

```python
# app/utils/logger.py
import logging
import sys
from pythonjsonlogger import jsonlogger

def setup_logging(name: str) -> logging.Logger:
    """Setup structured JSON logging"""
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    
    # JSON formatter
    formatter = jsonlogger.JsonFormatter(
        "%(timestamp)s %(level)s %(name)s %(message)s",
        rename_fields={"timestamp": "@timestamp"}
    )
    
    # Console handler
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    
    return logger

# Usage
logger = setup_logging(__name__)
logger.info("API started", extra={"version": "1.0.0", "environment": "production"})
```

### 15.2 Metrics Collection

```python
# app/core/metrics.py
from prometheus_client import Counter, Histogram, Gauge, generate_latest

# Define metrics
request_count = Counter(
    'http_requests_total',
    'Total HTTP requests',
    ['method', 'endpoint', 'status']
)

request_duration = Histogram(
    'http_request_duration_seconds',
    'HTTP request duration',
    ['method', 'endpoint']
)

active_videos = Gauge(
    'active_videos_processing',
    'Number of videos currently processing'
)

cost_per_video = Histogram(
    'cost_per_video_dollars',
    'Cost per video in dollars',
    buckets=[0.5, 1.0, 1.5, 2.0, 2.5, 3.0]
)

youtube_quota_usage = Gauge(
    'youtube_quota_usage',
    'YouTube API quota usage',
    ['account_email']
)

def get_metrics():
    """Export metrics in Prometheus format"""
    return generate_latest()
```

### 15.3 Health Checks

```python
# app/api/v1/endpoints/health.py
from fastapi import APIRouter, Depends
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import text
import redis.asyncio as redis

from app.database import get_db
from app.config import settings

router = APIRouter(prefix="/health", tags=["Health"])

@router.get("/")
async def health_check():
    """Basic health check"""
    return {"status": "healthy"}

@router.get("/detailed")
async def detailed_health_check(db: AsyncSession = Depends(get_db)):
    """Detailed health check with component status"""
    health = {
        "status": "healthy",
        "components": {}
    }
    
    # Check database
    try:
        await db.execute(text("SELECT 1"))
        health["components"]["database"] = "healthy"
    except Exception as e:
        health["components"]["database"] = f"unhealthy: {str(e)}"
        health["status"] = "degraded"
    
    # Check Redis
    try:
        redis_client = await redis.from_url(settings.REDIS_URL)
        await redis_client.ping()
        health["components"]["redis"] = "healthy"
    except Exception as e:
        health["components"]["redis"] = f"unhealthy: {str(e)}"
        health["status"] = "degraded"
    
    # Check YouTube accounts
    try:
        active_accounts = await db.scalar(
            select(func.count()).select_from(YouTubeAccount)
            .where(YouTubeAccount.is_active == True)
        )
        health["components"]["youtube_accounts"] = f"{active_accounts}/15 active"
    except Exception as e:
        health["components"]["youtube_accounts"] = f"error: {str(e)}"
    
    return health
```

---

## Quick Reference

### API Endpoints Summary

| Method | Endpoint | Description | Auth Required |
|--------|----------|-------------|---------------|
| POST | `/api/v1/auth/register` | Register new user | No |
| POST | `/api/v1/auth/login` | Login user | No |
| GET | `/api/v1/auth/me` | Get current user | Yes |
| GET | `/api/v1/channels` | List channels | Yes |
| POST | `/api/v1/channels` | Create channel | Yes |
| GET | `/api/v1/channels/{id}` | Get channel | Yes |
| PATCH | `/api/v1/channels/{id}` | Update channel | Yes |
| DELETE | `/api/v1/channels/{id}` | Delete channel | Yes |
| POST | `/api/v1/videos/generate` | Generate video | Yes |
| GET | `/api/v1/videos/{id}` | Get video | Yes |
| GET | `/api/v1/costs/summary` | Get cost summary | Yes |
| GET | `/api/v1/youtube/oauth/authorize` | Start OAuth | Yes |
| GET | `/api/v1/youtube/oauth/callback` | OAuth callback | No |
| POST | `/api/v1/webhooks/n8n/video-complete` | N8N completion | No |
| GET | `/api/v1/health` | Health check | No |
| GET | `/api/v1/metrics` | Prometheus metrics | No |

### Response Time Requirements

| Endpoint Type | P50 | P95 | P99 |
|--------------|-----|-----|-----|
| Read Operations | 50ms | 200ms | 500ms |
| Write Operations | 100ms | 500ms | 1000ms |
| Video Generation | N/A | N/A | 10min |
| YouTube Upload | N/A | N/A | 2min |

### Cost Limits

| Level | Amount | Action |
|-------|--------|--------|
| Target | $1.00 | Optimal cost per video |
| Warning | $2.50 | Switch to economy mode |
| Hard Limit | $3.00 | Stop processing immediately |
| Daily Budget | $150.00 | Pause all operations |

### YouTube Account Distribution

| Account Type | Count | Daily Videos | Purpose |
|-------------|-------|--------------|---------|
| Active | 12 | 5 per account | Primary uploads |
| Reserve | 3 | 2 per account | Emergency/overflow |
| Total | 15 | 66 max | Full capacity |

---

## Troubleshooting Guide

### Common Issues and Solutions

#### 1. Database Connection Issues

```python
# Error: connection to server at "localhost", port 5432 failed
# Solution: Check PostgreSQL is running
sudo systemctl status postgresql
sudo systemctl start postgresql

# Verify connection
psql -U ytempire_user -d ytempire_db -h localhost
```

#### 2. Redis Connection Issues

```python
# Error: Error connecting to Redis
# Solution: Check Redis is running
redis-cli ping

# Start Redis if needed
sudo systemctl start redis-server
```

#### 3. YouTube Quota Exceeded

```python
# Error: quotaExceeded
# Solution: Implement account rotation
async def handle_quota_exceeded(account_id: str):
    # Mark account as exhausted
    account = await db.get(YouTubeAccount, account_id)
    account.quota_used_today = 10000
    
    # Get next available account
    next_account = await youtube_service.get_best_account(db)
    
    if not next_account:
        # All accounts exhausted - wait until reset
        raise Exception("All YouTube accounts exhausted. Wait until midnight PST.")
    
    return next_account
```

#### 4. Cost Limit Exceeded

```python
# Error: Cost limit exceeded
# Solution: Switch to economy mode
async def switch_to_economy_mode(video_id: str):
    video = await db.get(Video, video_id)
    video.optimization_level = "economy"
    
    # Use cheaper services
    config = {
        "script_model": "gpt-3.5-turbo",  # Instead of GPT-4
        "tts_service": "google",           # Instead of ElevenLabs
        "video_quality": "720p",           # Instead of 1080p
        "effects": "minimal"               # Reduce processing
    }
    
    return config
```

#### 5. Slow API Response Times

```python
# Problem: API responses > 500ms
# Solution: Implement caching
@cache_result(ttl=300)
async def get_channel_stats(channel_id: str):
    # Expensive query - cache for 5 minutes
    stats = await db.execute(
        select(
            func.count(Video.id),
            func.sum(Video.total_cost),
            func.avg(Video.quality_score)
        ).where(Video.channel_id == channel_id)
    )
    return stats

# Add database indexes
CREATE INDEX CONCURRENTLY idx_videos_channel_status 
ON videos(channel_id, status) 
WHERE status IN ('queued', 'processing');
```

---

## Best Practices

### 1. Error Handling

```python
# Always use specific exception handling
from fastapi import HTTPException

async def get_video(video_id: str, db: AsyncSession):
    try:
        video = await db.get(Video, video_id)
        
        if not video:
            raise HTTPException(
                status_code=404,
                detail=f"Video {video_id} not found"
            )
        
        return video
        
    except HTTPException:
        raise  # Re-raise HTTP exceptions
        
    except Exception as e:
        logger.error(f"Error fetching video {video_id}: {e}")
        raise HTTPException(
            status_code=500,
            detail="Internal server error"
        )
```

### 2. Database Transactions

```python
# Use transactions for multi-step operations
async def transfer_channel_ownership(
    channel_id: str,
    new_user_id: str,
    db: AsyncSession
):
    async with db.begin():  # Automatic rollback on error
        channel = await db.get(Channel, channel_id)
        old_user = await db.get(User, channel.user_id)
        new_user = await db.get(User, new_user_id)
        
        # Update limits
        old_user.channel_limit += 1
        new_user.channel_limit -= 1
        
        # Transfer ownership
        channel.user_id = new_user_id
        
        # Commit happens automatically
```

### 3. Async Best Practices

```python
# Use asyncio.gather for parallel operations
async def get_dashboard_data(user_id: str):
    # Run queries in parallel
    channels, videos, costs = await asyncio.gather(
        get_user_channels(user_id),
        get_recent_videos(user_id),
        get_cost_summary(user_id)
    )
    
    return {
        "channels": channels,
        "videos": videos,
        "costs": costs
    }

# Don't block the event loop
# Bad: time.sleep(5)
# Good: await asyncio.sleep(5)
```

### 4. Security Best Practices

```python
# Always validate input
from pydantic import validator

class VideoCreate(BaseModel):
    title: str
    description: str
    
    @validator('title')
    def validate_title(cls, v):
        if len(v) > 100:
            raise ValueError('Title too long')
        if not v.strip():
            raise ValueError('Title cannot be empty')
        return v.strip()

# Sanitize user input for SQL
# Use parameterized queries (SQLAlchemy does this automatically)
# Never: f"SELECT * FROM users WHERE email = '{email}'"
# Always: select(User).where(User.email == email)

# Rate limiting
from slowapi import Limiter
limiter = Limiter(key_func=get_remote_address)

@app.get("/api/v1/expensive-operation")
@limiter.limit("5/minute")
async def expensive_operation(request: Request):
    return {"result": "success"}
```

### 5. Logging Best Practices

```python
# Use structured logging with context
logger.info(
    "Video processing completed",
    extra={
        "video_id": video_id,
        "channel_id": channel_id,
        "processing_time": processing_time,
        "cost": total_cost,
        "status": "success"
    }
)

# Log at appropriate levels
logger.debug("Detailed debug information")
logger.info("Normal operations")
logger.warning("Warning conditions")
logger.error("Error conditions")
logger.critical("System failures")
```

---

## Performance Optimization Checklist

### Database Optimizations

- [ ] Add indexes on frequently queried columns
- [ ] Use connection pooling (configured in SQLAlchemy)
- [ ] Implement query result caching
- [ ] Use eager loading for relationships
- [ ] Limit SELECT columns to needed fields only
- [ ] Use database views for complex queries
- [ ] Implement pagination for list endpoints
- [ ] Regular VACUUM and ANALYZE operations

### API Optimizations

- [ ] Implement response caching with Redis
- [ ] Use async/await throughout
- [ ] Batch operations where possible
- [ ] Implement request/response compression
- [ ] Use CDN for static assets
- [ ] Implement rate limiting
- [ ] Use connection pooling for external services
- [ ] Implement circuit breakers for external APIs

### Cost Optimizations

- [ ] Cache OpenAI responses
- [ ] Use GPT-3.5 instead of GPT-4 when possible
- [ ] Batch TTS requests
- [ ] Implement quota pooling for YouTube accounts
- [ ] Monitor and alert on cost thresholds
- [ ] Use webhooks instead of polling
- [ ] Implement gradual quality degradation
- [ ] Pre-calculate and cache expensive metrics

---

## Security Checklist

### Authentication & Authorization

- [ ] JWT tokens with expiration
- [ ] Refresh token rotation
- [ ] Password complexity requirements
- [ ] Account lockout after failed attempts
- [ ] Two-factor authentication (future)
- [ ] API key management for services
- [ ] Role-based access control
- [ ] Session invalidation on password change

### Data Protection

- [ ] HTTPS only in production
- [ ] Encrypt sensitive data at rest
- [ ] Sanitize all user inputs
- [ ] SQL injection prevention (parameterized queries)
- [ ] XSS prevention (output encoding)
- [ ] CSRF tokens for state-changing operations
- [ ] Secure password hashing (bcrypt)
- [ ] PII data encryption

### Infrastructure Security

- [ ] Environment variables for secrets
- [ ] Regular security updates
- [ ] Firewall configuration
- [ ] Database access restrictions
- [ ] Redis password protection
- [ ] Log sensitive data masking
- [ ] Regular security audits
- [ ] Backup encryption

---

## Appendix A: Environment Variables

### Complete .env Template

```bash
# Application Configuration
PROJECT_NAME=YTEMPIRE
ENVIRONMENT=development
SECRET_KEY=your-super-secret-key-minimum-32-characters-long
API_V1_STR=/api/v1
DEBUG=False

# Database Configuration
POSTGRES_SERVER=localhost
POSTGRES_USER=ytempire_user
POSTGRES_PASSWORD=your_postgres_password
POSTGRES_DB=ytempire_db
POSTGRES_PORT=5432

# Redis Configuration
REDIS_HOST=localhost
REDIS_PORT=6379
REDIS_DB=0
REDIS_PASSWORD=your_redis_password

# JWT Configuration
ALGORITHM=HS256
ACCESS_TOKEN_EXPIRE_MINUTES=30
REFRESH_TOKEN_EXPIRE_DAYS=7

# YouTube API Configuration
YOUTUBE_CLIENT_ID=your-client-id.apps.googleusercontent.com
YOUTUBE_CLIENT_SECRET=your-youtube-client-secret
YOUTUBE_REDIRECT_URI=http://localhost:8000/api/v1/auth/youtube/callback
YOUTUBE_SCOPES=https://www.googleapis.com/auth/youtube.upload,https://www.googleapis.com/auth/youtube

# OpenAI Configuration
OPENAI_API_KEY=sk-your-openai-api-key
OPENAI_MODEL=gpt-3.5-turbo
OPENAI_MAX_TOKENS=2000
OPENAI_TEMPERATURE=0.7

# Google Text-to-Speech
GOOGLE_APPLICATION_CREDENTIALS=/path/to/google-credentials.json
GOOGLE_TTS_LANGUAGE=en-US
GOOGLE_TTS_VOICE=en-US-Neural2-J

# ElevenLabs Configuration
ELEVENLABS_API_KEY=your-elevenlabs-api-key
ELEVENLABS_VOICE_ID=21m00Tcm4TlvDq8ikWAM

# Stripe Configuration
STRIPE_SECRET_KEY=sk_test_your-stripe-secret-key
STRIPE_WEBHOOK_SECRET=whsec_your-webhook-secret
STRIPE_PRICE_ID_STARTER=price_starter_monthly
STRIPE_PRICE_ID_GROWTH=price_growth_monthly
STRIPE_PRICE_ID_SCALE=price_scale_monthly

# N8N Configuration
N8N_BASE_URL=http://localhost:5678
N8N_API_KEY=your-n8n-api-key
N8N_WEBHOOK_URL=http://localhost:8000/api/v1/webhooks/n8n

# Cost Management
MAX_COST_PER_VIDEO=3.00
WARNING_COST_THRESHOLD=2.50
TARGET_COST_PER_VIDEO=1.00
DAILY_COST_BUDGET=150.00
MONTHLY_COST_BUDGET=4500.00

# Rate Limiting
RATE_LIMIT_PER_MINUTE=60
RATE_LIMIT_PER_HOUR=1000
RATE_LIMIT_PER_DAY=10000

# File Storage
STORAGE_BASE_PATH=/mnt/nvme/ytempire
MAX_FILE_SIZE_MB=1000
VIDEO_RETENTION_DAYS=7
THUMBNAIL_RETENTION_DAYS=30

# Monitoring
ENABLE_METRICS=True
ENABLE_TRACING=False
LOG_LEVEL=INFO
SENTRY_DSN=https://your-sentry-dsn@sentry.io/project-id

# Email Configuration (Future)
SMTP_HOST=smtp.gmail.com
SMTP_PORT=587
SMTP_USER=your-email@gmail.com
SMTP_PASSWORD=your-app-password
FROM_EMAIL=noreply@ytempire.com

# CORS Settings
BACKEND_CORS_ORIGINS=["http://localhost:3000","http://localhost:8000"]
ALLOWED_HOSTS=["localhost","127.0.0.1","ytempire.com"]
```

---

## Appendix B: Common SQL Queries

### Useful Queries for Monitoring

```sql
-- Daily video generation stats
SELECT 
    DATE(created_at) as date,
    COUNT(*) as videos_created,
    COUNT(CASE WHEN status = 'published' THEN 1 END) as published,
    COUNT(CASE WHEN status = 'failed' THEN 1 END) as failed,
    AVG(total_cost) as avg_cost,
    SUM(total_cost) as total_cost
FROM videos
WHERE created_at >= CURRENT_DATE - INTERVAL '7 days'
GROUP BY DATE(created_at)
ORDER BY date DESC;

-- YouTube account health
SELECT 
    email,
    is_active,
    is_reserve,
    health_score,
    uploads_today,
    quota_used_today,
    consecutive_errors,
    last_upload_at,
    last_error_at
FROM youtube_accounts
ORDER BY health_score DESC;

-- Channel performance
SELECT 
    c.name,
    c.niche,
    COUNT(v.id) as total_videos,
    AVG(v.total_cost) as avg_cost,
    c.total_views,
    c.total_subscribers,
    c.total_revenue_cents / 100.0 as revenue_dollars
FROM channels c
LEFT JOIN videos v ON c.id = v.channel_id
GROUP BY c.id
ORDER BY revenue_dollars DESC;

-- Cost analysis by service
SELECT 
    service,
    COUNT(*) as usage_count,
    SUM(amount) as total_cost,
    AVG(amount) as avg_cost,
    MAX(amount) as max_cost
FROM video_costs
WHERE timestamp >= CURRENT_DATE - INTERVAL '24 hours'
GROUP BY service
ORDER BY total_cost DESC;
```

---

## Appendix C: API Response Examples

### Successful Responses

```json
// GET /api/v1/channels
{
  "items": [
    {
      "id": "550e8400-e29b-41d4-a716-446655440000",
      "name": "Tech Reviews",
      "niche": "Technology",
      "status": "active",
      "automation_enabled": true,
      "total_videos": 150,
      "total_views": 50000,
      "created_at": "2025-01-15T10:30:00Z"
    }
  ],
  "total": 5,
  "page": 1,
  "pages": 1,
  "limit": 10
}

// POST /api/v1/videos/generate
{
  "id": "660e8400-e29b-41d4-a716-446655440001",
  "channel_id": "550e8400-e29b-41d4-a716-446655440000",
  "title": "Top 10 Gadgets of 2025",
  "status": "queued",
  "priority": 5,
  "estimated_cost": 2.50,
  "created_at": "2025-01-15T14:20:00Z",
  "estimated_completion": "2025-01-15T14:30:00Z"
}
```

### Error Responses

```json
// 400 Bad Request
{
  "detail": "Invalid request parameters",
  "errors": [
    {
      "field": "title",
      "message": "Title cannot be empty"
    }
  ]
}

// 401 Unauthorized
{
  "detail": "Could not validate credentials"
}

// 403 Forbidden
{
  "detail": "Channel limit reached (5). Upgrade to Growth plan for more channels."
}

// 404 Not Found
{
  "detail": "Channel not found"
}

// 429 Too Many Requests
{
  "detail": "Rate limit exceeded",
  "retry_after": 60
}

// 500 Internal Server Error
{
  "detail": "Internal server error",
  "request_id": "550e8400-e29b-41d4-a716-446655440000"
}
```

---

## Final Notes

This comprehensive guide provides everything you need to implement the YTEMPIRE backend APIs. Remember:

1. **Start with the basics**: Get authentication and basic CRUD operations working first
2. **Test as you go**: Write tests for each endpoint as you implement them
3. **Monitor costs closely**: The $3/video limit is a hard constraint
4. **Optimize gradually**: Don't premature optimize, measure first
5. **Document changes**: Keep this guide updated as the system evolves

For questions or clarifications, refer to the team documentation or reach out to the Backend Team Lead.

**Good luck building the future of automated content creation!** 🚀