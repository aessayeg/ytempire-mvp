# YTEMPIRE API Development Engineer Documentation
**Version 1.0 | January 2025**  
**Role: API Development Engineer**  
**Reports to: Backend Team Lead**  
**Document Status: MVP Implementation Guide**

---

## Executive Summary

Welcome to the YTEMPIRE API Development team! As an API Development Engineer, you'll be building the core backend infrastructure that powers our revolutionary YouTube automation platform. This document provides comprehensive guidance for implementing the API layer that will serve 50 beta users managing 250+ YouTube channels with 95% automation.

**Your Mission**: Build robust, scalable APIs that enable users to generate $10,000+ monthly revenue with just 1 hour of weekly oversight across 5+ YouTube channels.

---

## 1. Your Role in the Organization

### 1.1 Team Structure
```
CTO/Technical Director
        │
        ├── Backend Team Lead (Your Manager)
        │   ├── API Developer Engineer (You)
        │   ├── Data Pipeline Engineer
        │   └── Integration Specialist
        │
        ├── Frontend Team Lead
        └── Platform Ops Lead
```

### 1.2 Key Collaborations
- **Data Pipeline Engineer**: Coordinate on video processing queue design
- **Integration Specialist**: Align on YouTube API and third-party integrations
- **Frontend Team**: Define API contracts and real-time update mechanisms
- **Platform Ops**: Ensure API performance meets SLOs

### 1.3 Core Responsibilities
1. Design and implement RESTful APIs for all platform features
2. Build authentication and authorization systems
3. Implement cost tracking and optimization logic
4. Create robust error handling and retry mechanisms
5. Ensure API performance meets <500ms p95 latency requirement

---

## 2. Technical Architecture Overview

### 2.1 System Architecture
```yaml
architecture:
  type: Monolithic API (MVP) → Microservices (Future)
  framework: FastAPI (Python 3.11+)
  database: PostgreSQL 15
  cache: Redis 7
  queue: Redis + N8N
  storage: Local NVMe (4TB for videos)
  
  deployment:
    environment: Local server (MVP)
    containerization: Docker + Docker Compose
    reverse_proxy: Nginx
    process_manager: Gunicorn with Uvicorn workers
```

### 2.2 API Architecture Principles
```python
# Core principles for all API development

class APIDesignPrinciples:
    """YTEMPIRE API Development Standards"""
    
    PATTERNS = {
        "consistency": "All endpoints follow RESTful conventions",
        "versioning": "API v1 with clear upgrade path",
        "pagination": "Cursor-based for scalability",
        "filtering": "Standardized query parameters",
        "sorting": "Multi-field with direction control",
        "responses": "Consistent JSON structure with metadata"
    }
    
    PERFORMANCE = {
        "latency_target": "p95 < 500ms",
        "connection_pooling": "Required for all services",
        "caching_strategy": "Redis for hot paths",
        "batch_operations": "Support bulk endpoints",
        "rate_limiting": "Per-user and per-endpoint"
    }
    
    RELIABILITY = {
        "error_handling": "Graceful degradation",
        "retry_logic": "Exponential backoff",
        "circuit_breakers": "Prevent cascade failures",
        "health_checks": "Detailed service status",
        "monitoring": "Prometheus metrics on all endpoints"
    }
```

### 2.3 Technology Stack
```yaml
core_stack:
  language: Python 3.11+
  framework: FastAPI 0.104+
  orm: SQLAlchemy 2.0+
  validation: Pydantic v2
  authentication: JWT + OAuth2
  
dependencies:
  database:
    - psycopg2-binary
    - alembic (migrations)
  
  caching:
    - redis-py
    - fakeredis (testing)
  
  external_apis:
    - google-api-python-client (YouTube)
    - openai (GPT-4)
    - elevenlabs (voice synthesis)
  
  utilities:
    - httpx (async HTTP)
    - python-multipart (file uploads)
    - python-jose (JWT)
    - passlib (password hashing)
    - celery (future - background tasks)
  
  monitoring:
    - prometheus-client
    - opentelemetry-api
    - structlog
```

---

## 3. API Module Structure

### 3.1 Project Structure
```
ytempire-backend/
├── app/
│   ├── __init__.py
│   ├── main.py              # FastAPI application entry
│   ├── config.py            # Configuration management
│   ├── database.py          # Database connection setup
│   ├── dependencies.py      # Shared dependencies
│   │
│   ├── api/
│   │   ├── __init__.py
│   │   ├── v1/
│   │   │   ├── __init__.py
│   │   │   ├── auth.py      # Authentication endpoints
│   │   │   ├── users.py     # User management
│   │   │   ├── channels.py  # Channel operations
│   │   │   ├── videos.py    # Video generation/management
│   │   │   ├── analytics.py # Analytics endpoints
│   │   │   ├── webhooks.py  # External webhooks
│   │   │   └── admin.py     # Admin operations
│   │   └── deps.py          # API dependencies
│   │
│   ├── core/
│   │   ├── __init__.py
│   │   ├── security.py      # Security utilities
│   │   ├── config.py        # Core configuration
│   │   ├── exceptions.py    # Custom exceptions
│   │   └── utils.py         # Utility functions
│   │
│   ├── models/
│   │   ├── __init__.py
│   │   ├── user.py          # User model
│   │   ├── channel.py       # Channel model
│   │   ├── video.py         # Video model
│   │   ├── analytics.py     # Analytics models
│   │   └── base.py          # Base model class
│   │
│   ├── schemas/
│   │   ├── __init__.py
│   │   ├── user.py          # User schemas
│   │   ├── channel.py       # Channel schemas
│   │   ├── video.py         # Video schemas
│   │   ├── analytics.py     # Analytics schemas
│   │   └── common.py        # Common schemas
│   │
│   ├── services/
│   │   ├── __init__.py
│   │   ├── auth.py          # Authentication service
│   │   ├── youtube.py       # YouTube API integration
│   │   ├── ai_content.py    # AI content generation
│   │   ├── video_processor.py # Video processing
│   │   ├── cost_tracker.py  # Cost management
│   │   └── analytics.py     # Analytics service
│   │
│   └── workers/
│       ├── __init__.py
│       └── tasks.py         # Background tasks
│
├── migrations/              # Alembic migrations
├── tests/                   # Test suite
├── scripts/                 # Utility scripts
├── docker-compose.yml       # Local development
├── Dockerfile              # Container definition
├── requirements.txt        # Dependencies
└── .env.example           # Environment template
```

### 3.2 Core API Endpoints

```python
# API Route Structure
API_V1_PREFIX = "/api/v1"

ENDPOINTS = {
    # Authentication
    "POST /auth/register": "User registration",
    "POST /auth/login": "User login",
    "POST /auth/refresh": "Refresh JWT token",
    "POST /auth/logout": "User logout",
    
    # User Management
    "GET /users/me": "Get current user",
    "PUT /users/me": "Update user profile",
    "GET /users/me/usage": "Get usage statistics",
    "GET /users/me/billing": "Get billing information",
    
    # Channel Management
    "GET /channels": "List user channels",
    "POST /channels": "Create new channel",
    "GET /channels/{id}": "Get channel details",
    "PUT /channels/{id}": "Update channel",
    "DELETE /channels/{id}": "Delete channel",
    "POST /channels/{id}/sync": "Sync with YouTube",
    
    # Video Operations
    "GET /videos": "List videos with filters",
    "POST /videos/generate": "Generate new video",
    "GET /videos/{id}": "Get video details",
    "PUT /videos/{id}": "Update video metadata",
    "DELETE /videos/{id}": "Delete video",
    "POST /videos/{id}/publish": "Publish to YouTube",
    "GET /videos/{id}/cost": "Get video cost breakdown",
    
    # Analytics
    "GET /analytics/overview": "Dashboard overview",
    "GET /analytics/channels/{id}": "Channel analytics",
    "GET /analytics/videos/{id}": "Video analytics",
    "GET /analytics/revenue": "Revenue analytics",
    
    # Webhooks
    "POST /webhooks/youtube": "YouTube notifications",
    "POST /webhooks/stripe": "Payment notifications",
    
    # Admin (if authorized)
    "GET /admin/users": "List all users",
    "GET /admin/system/health": "System health check",
    "GET /admin/system/metrics": "System metrics"
}
```

---

## 4. Database Design

### 4.1 Core Models
```python
# SQLAlchemy Models Structure

from sqlalchemy import Column, String, Integer, DateTime, ForeignKey, JSON, Numeric, Boolean
from sqlalchemy.orm import relationship
from sqlalchemy.ext.declarative import declarative_base

Base = declarative_base()

class User(Base):
    __tablename__ = "users"
    
    id = Column(Integer, primary_key=True)
    email = Column(String(255), unique=True, nullable=False, index=True)
    username = Column(String(100), unique=True, nullable=False)
    hashed_password = Column(String(255), nullable=False)
    is_active = Column(Boolean, default=True)
    is_superuser = Column(Boolean, default=False)
    created_at = Column(DateTime, nullable=False)
    updated_at = Column(DateTime, nullable=False)
    
    # Relationships
    channels = relationship("Channel", back_populates="user")
    api_keys = relationship("APIKey", back_populates="user")
    usage_stats = relationship("UsageStats", back_populates="user")

class Channel(Base):
    __tablename__ = "channels"
    
    id = Column(Integer, primary_key=True)
    user_id = Column(Integer, ForeignKey("users.id"), nullable=False)
    youtube_channel_id = Column(String(255), unique=True, nullable=False)
    title = Column(String(255), nullable=False)
    description = Column(Text)
    niche = Column(String(100), nullable=False)
    status = Column(String(50), default="active")
    youtube_credentials = Column(JSON)  # Encrypted
    settings = Column(JSON)
    created_at = Column(DateTime, nullable=False)
    updated_at = Column(DateTime, nullable=False)
    
    # Relationships
    user = relationship("User", back_populates="channels")
    videos = relationship("Video", back_populates="channel")
    analytics = relationship("ChannelAnalytics", back_populates="channel")

class Video(Base):
    __tablename__ = "videos"
    
    id = Column(Integer, primary_key=True)
    channel_id = Column(Integer, ForeignKey("channels.id"), nullable=False)
    youtube_video_id = Column(String(255), unique=True)
    title = Column(String(255), nullable=False)
    description = Column(Text)
    script = Column(Text)
    status = Column(String(50), default="draft")  # draft, processing, published, failed
    generation_params = Column(JSON)
    file_path = Column(String(500))
    thumbnail_path = Column(String(500))
    duration_seconds = Column(Integer)
    cost_breakdown = Column(JSON)
    total_cost = Column(Numeric(10, 2))
    published_at = Column(DateTime)
    created_at = Column(DateTime, nullable=False)
    updated_at = Column(DateTime, nullable=False)
    
    # Relationships
    channel = relationship("Channel", back_populates="videos")
    analytics = relationship("VideoAnalytics", back_populates="video")
    generation_logs = relationship("GenerationLog", back_populates="video")
```

### 4.2 Database Optimization
```yaml
optimization_strategies:
  indexes:
    - users.email (unique)
    - users.username (unique)
    - channels.youtube_channel_id (unique)
    - channels.user_id
    - videos.channel_id
    - videos.status
    - videos.created_at
    
  partitioning:
    videos: By created_at (monthly)
    analytics: By date (daily)
    logs: By timestamp (weekly)
    
  connection_pooling:
    pool_size: 20
    max_overflow: 10
    pool_timeout: 30
    pool_recycle: 3600
```

---

## 5. Core Implementation Examples

### 5.1 FastAPI Application Setup
```python
# app/main.py
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from prometheus_fastapi_instrumentator import Instrumentator
import structlog

from app.api.v1 import auth, users, channels, videos, analytics
from app.core.config import settings
from app.database import engine
from app.models import Base

# Configure structured logging
structlog.configure(
    processors=[
        structlog.stdlib.filter_by_level,
        structlog.stdlib.add_logger_name,
        structlog.stdlib.add_log_level,
        structlog.stdlib.PositionalArgumentsFormatter(),
        structlog.processors.TimeStamper(fmt="iso"),
        structlog.processors.StackInfoRenderer(),
        structlog.processors.format_exc_info,
        structlog.processors.UnicodeDecoder(),
        structlog.processors.JSONRenderer()
    ],
    context_class=dict,
    logger_factory=structlog.stdlib.LoggerFactory(),
    cache_logger_on_first_use=True,
)

logger = structlog.get_logger()

# Create FastAPI app
app = FastAPI(
    title="YTEMPIRE API",
    description="Automated YouTube Empire Building Platform",
    version="1.0.0",
    docs_url="/api/docs",
    redoc_url="/api/redoc"
)

# CORS configuration
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.BACKEND_CORS_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Prometheus metrics
Instrumentator().instrument(app).expose(app)

# Include routers
app.include_router(auth.router, prefix="/api/v1/auth", tags=["authentication"])
app.include_router(users.router, prefix="/api/v1/users", tags=["users"])
app.include_router(channels.router, prefix="/api/v1/channels", tags=["channels"])
app.include_router(videos.router, prefix="/api/v1/videos", tags=["videos"])
app.include_router(analytics.router, prefix="/api/v1/analytics", tags=["analytics"])

@app.on_event("startup")
async def startup_event():
    """Initialize services on startup"""
    logger.info("Starting YTEMPIRE API", version="1.0.0")
    # Create database tables
    Base.metadata.create_all(bind=engine)
    # Initialize Redis connection
    # Initialize external service clients
    
@app.on_event("shutdown")
async def shutdown_event():
    """Cleanup on shutdown"""
    logger.info("Shutting down YTEMPIRE API")
    # Close database connections
    # Close Redis connections
    # Cleanup temporary files

@app.get("/health")
async def health_check():
    """Health check endpoint for Platform Ops monitoring"""
    return {
        "status": "healthy",
        "version": "1.0.0",
        "timestamp": datetime.utcnow().isoformat()
    }
```

### 5.2 Authentication Implementation
```python
# app/api/v1/auth.py
from datetime import datetime, timedelta
from typing import Optional

from fastapi import APIRouter, Depends, HTTPException, status
from fastapi.security import OAuth2PasswordRequestForm
from jose import JWTError, jwt
from passlib.context import CryptContext
from sqlalchemy.orm import Session

from app.core.config import settings
from app.database import get_db
from app.models import User
from app.schemas.auth import Token, TokenData
from app.schemas.user import UserCreate, UserResponse

router = APIRouter()
pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")

def create_access_token(data: dict, expires_delta: Optional[timedelta] = None):
    to_encode = data.copy()
    if expires_delta:
        expire = datetime.utcnow() + expires_delta
    else:
        expire = datetime.utcnow() + timedelta(minutes=settings.ACCESS_TOKEN_EXPIRE_MINUTES)
    to_encode.update({"exp": expire})
    encoded_jwt = jwt.encode(to_encode, settings.SECRET_KEY, algorithm=settings.ALGORITHM)
    return encoded_jwt

@router.post("/register", response_model=UserResponse)
async def register(
    user_data: UserCreate,
    db: Session = Depends(get_db)
):
    """Register a new user"""
    # Check if user exists
    if db.query(User).filter(User.email == user_data.email).first():
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Email already registered"
        )
    
    # Create new user
    hashed_password = pwd_context.hash(user_data.password)
    db_user = User(
        email=user_data.email,
        username=user_data.username,
        hashed_password=hashed_password,
        created_at=datetime.utcnow(),
        updated_at=datetime.utcnow()
    )
    
    db.add(db_user)
    db.commit()
    db.refresh(db_user)
    
    # Log registration
    logger.info("New user registered", user_id=db_user.id, email=db_user.email)
    
    return db_user

@router.post("/login", response_model=Token)
async def login(
    form_data: OAuth2PasswordRequestForm = Depends(),
    db: Session = Depends(get_db)
):
    """Login and receive access token"""
    user = authenticate_user(db, form_data.username, form_data.password)
    if not user:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Incorrect username or password",
            headers={"WWW-Authenticate": "Bearer"},
        )
    
    access_token_expires = timedelta(minutes=settings.ACCESS_TOKEN_EXPIRE_MINUTES)
    access_token = create_access_token(
        data={"sub": user.email}, expires_delta=access_token_expires
    )
    
    # Track login
    logger.info("User logged in", user_id=user.id)
    
    return {"access_token": access_token, "token_type": "bearer"}
```

---

## 6. External API Integration

### 6.1 YouTube API Integration
```python
# app/services/youtube.py
from google.oauth2.credentials import Credentials
from googleapiclient.discovery import build
from googleapiclient.http import MediaFileUpload
import asyncio
from typing import Dict, Optional

class YouTubeService:
    """YouTube API integration service"""
    
    def __init__(self):
        self.quota_limit = 10000  # Daily quota
        self.quota_used = 0
        self.api_version = "v3"
        
    async def initialize_client(self, credentials: Dict) -> Any:
        """Initialize YouTube API client with user credentials"""
        creds = Credentials(
            token=credentials['access_token'],
            refresh_token=credentials['refresh_token'],
            token_uri="https://oauth2.googleapis.com/token",
            client_id=settings.YOUTUBE_CLIENT_ID,
            client_secret=settings.YOUTUBE_CLIENT_SECRET
        )
        
        return build('youtube', self.api_version, credentials=creds)
    
    async def upload_video(
        self, 
        channel_id: str,
        video_path: str,
        title: str,
        description: str,
        tags: List[str],
        category_id: str = "22"  # People & Blogs
    ) -> Dict:
        """Upload video to YouTube channel"""
        
        # Check quota
        if self.quota_used + 1600 > self.quota_limit:  # Upload costs ~1600 quota
            raise Exception("YouTube API quota exceeded")
        
        try:
            youtube = await self.initialize_client(channel_credentials)
            
            body = {
                'snippet': {
                    'title': title,
                    'description': description,
                    'tags': tags,
                    'categoryId': category_id,
                },
                'status': {
                    'privacyStatus': 'private',  # Start as private
                    'selfDeclaredMadeForKids': False
                }
            }
            
            # Call the API's videos.insert method
            media = MediaFileUpload(video_path, chunksize=-1, resumable=True)
            
            request = youtube.videos().insert(
                part=",".join(body.keys()),
                body=body,
                media_body=media
            )
            
            response = await asyncio.to_thread(request.execute)
            
            self.quota_used += 1600
            logger.info("Video uploaded", video_id=response['id'], channel_id=channel_id)
            
            return {
                'youtube_video_id': response['id'],
                'upload_status': 'success',
                'video_link': f"https://youtube.com/watch?v={response['id']}"
            }
            
        except Exception as e:
            logger.error("YouTube upload failed", error=str(e), channel_id=channel_id)
            raise
```

### 6.2 AI Content Generation Service
```python
# app/services/ai_content.py
import openai
from typing import Dict, List
import asyncio

class AIContentService:
    """AI-powered content generation service"""
    
    def __init__(self):
        openai.api_key = settings.OPENAI_API_KEY
        self.model = "gpt-3.5-turbo"  # Cost-effective for MVP
        self.cost_per_1k_tokens = 0.002
        
    async def generate_video_script(
        self,
        topic: str,
        video_length: int = 300,  # seconds
        style: str = "informative",
        niche: str = "general"
    ) -> Dict:
        """Generate video script using OpenAI"""
        
        # Calculate approximate words needed
        words_per_minute = 150
        target_words = (video_length / 60) * words_per_minute
        
        prompt = f"""
        Create a YouTube video script about: {topic}
        
        Requirements:
        - Length: approximately {target_words} words ({video_length} seconds when spoken)
        - Style: {style}
        - Niche: {niche}
        - Include an engaging hook in the first 10 seconds
        - Structure: Introduction, 3-5 main points, conclusion with call-to-action
        - Make it engaging and suitable for YouTube audience
        
        Format the output as:
        TITLE: [Compelling video title]
        DESCRIPTION: [YouTube description with timestamps]
        TAGS: [Comma-separated relevant tags]
        SCRIPT: [The actual script to be spoken]
        """
        
        try:
            response = await asyncio.to_thread(
                openai.ChatCompletion.create,
                model=self.model,
                messages=[
                    {"role": "system", "content": "You are an expert YouTube content creator."},
                    {"role": "user", "content": prompt}
                ],
                max_tokens=int(target_words * 1.5),
                temperature=0.7
            )
            
            content = response.choices[0].message.content
            
            # Parse response
            parsed = self._parse_script_response(content)
            
            # Calculate cost
            total_tokens = response.usage.total_tokens
            cost = (total_tokens / 1000) * self.cost_per_1k_tokens
            
            return {
                "script": parsed["script"],
                "title": parsed["title"],
                "description": parsed["description"],
                "tags": parsed["tags"],
                "cost": cost,
                "tokens_used": total_tokens
            }
            
        except Exception as e:
            logger.error("Script generation failed", error=str(e), topic=topic)
            raise
```

---

## 7. Performance & Monitoring

### 7.1 API Performance Optimization
```python
# app/core/performance.py
from functools import wraps
from typing import Callable
import time
import redis
from prometheus_client import Histogram, Counter

# Prometheus metrics
request_duration = Histogram(
    'api_request_duration_seconds',
    'API request duration',
    ['method', 'endpoint', 'status']
)

request_count = Counter(
    'api_request_count',
    'API request count',
    ['method', 'endpoint', 'status']
)

class PerformanceMonitor:
    """API performance monitoring"""
    
    def __init__(self):
        self.redis_client = redis.Redis(
            host=settings.REDIS_HOST,
            port=settings.REDIS_PORT,
            decode_responses=True
        )
    
    def track_request(self, func: Callable) -> Callable:
        """Decorator to track API request performance"""
        
        @wraps(func)
        async def wrapper(*args, **kwargs):
            start_time = time.time()
            
            try:
                result = await func(*args, **kwargs)
                status = "success"
                return result
                
            except Exception as e:
                status = "error"
                raise
                
            finally:
                duration = time.time() - start_time
                
                # Record metrics
                request_duration.labels(
                    method=kwargs.get('method', 'GET'),
                    endpoint=func.__name__,
                    status=status
                ).observe(duration)
                
                request_count.labels(
                    method=kwargs.get('method', 'GET'),
                    endpoint=func.__name__,
                    status=status
                ).inc()
                
                # Log slow requests
                if duration > 1.0:  # 1 second threshold
                    logger.warning(
                        "Slow API request",
                        endpoint=func.__name__,
                        duration=duration,
                        status=status
                    )
        
        return wrapper
    
    async def cache_response(self, key: str, data: Dict, ttl: int = 300):
        """Cache API response in Redis"""
        await self.redis_client.setex(
            key,
            ttl,
            json.dumps(data)
        )
    
    async def get_cached_response(self, key: str) -> Optional[Dict]:
        """Retrieve cached response"""
        data = await self.redis_client.get(key)
        return json.loads(data) if data else None
```

### 7.2 Rate Limiting Implementation
```python
# app/core/rate_limit.py
from fastapi import HTTPException, Request
from typing import Callable
import time

class RateLimiter:
    """API rate limiting implementation"""
    
    def __init__(
        self,
        requests_per_minute: int = 60,
        requests_per_hour: int = 1000
    ):
        self.rpm = requests_per_minute
        self.rph = requests_per_hour
        self.redis_client = redis.Redis(
            host=settings.REDIS_HOST,
            port=settings.REDIS_PORT
        )
    
    async def check_rate_limit(
        self,
        request: Request,
        user_id: int
    ) -> bool:
        """Check if user has exceeded rate limits"""
        
        current_minute = int(time.time() / 60)
        current_hour = int(time.time() / 3600)
        
        # Keys for tracking
        minute_key = f"rate_limit:minute:{user_id}:{current_minute}"
        hour_key = f"rate_limit:hour:{user_id}:{current_hour}"
        
        # Check minute limit
        minute_count = await self.redis_client.incr(minute_key)
        if minute_count == 1:
            await self.redis_client.expire(minute_key, 60)
        
        if minute_count > self.rpm:
            raise HTTPException(
                status_code=429,
                detail=f"Rate limit exceeded: {self.rpm} requests per minute"
            )
        
        # Check hour limit
        hour_count = await self.redis_client.incr(hour_key)
        if hour_count == 1:
            await self.redis_client.expire(hour_key, 3600)
        
        if hour_count > self.rph:
            raise HTTPException(
                status_code=429,
                detail=f"Rate limit exceeded: {self.rph} requests per hour"
            )
        
        return True
```

---

## 8. Testing Strategy

### 8.1 Unit Testing
```python
# tests/test_api/test_videos.py
import pytest
from httpx import AsyncClient
from sqlalchemy.orm import Session

from app.main import app
from app.models import User, Channel, Video
from tests.utils import create_test_user, create_test_channel

@pytest.mark.asyncio
async def test_create_video(
    client: AsyncClient,
    db: Session,
    test_user: User
):
    """Test video creation endpoint"""
    
    # Create test channel
    channel = create_test_channel(db, test_user.id)
    
    # Video creation payload
    payload = {
        "channel_id": channel.id,
        "topic": "10 Python Tips for Beginners",
        "video_length": 300,
        "style": "educational"
    }
    
    # Make request
    response = await client.post(
        "/api/v1/videos/generate",
        json=payload,
        headers={"Authorization": f"Bearer {test_user.token}"}
    )
    
    assert response.status_code == 201
    data = response.json()
    
    # Verify response
    assert "id" in data
    assert data["status"] == "processing"
    assert data["channel_id"] == channel.id
    
    # Verify database
    video = db.query(Video).filter(Video.id == data["id"]).first()
    assert video is not None
    assert video.title is not None
    assert video.total_cost < 1.0  # Under $1 target

@pytest.mark.asyncio
async def test_video_cost_tracking(
    client: AsyncClient,
    db: Session,
    test_video: Video
):
    """Test video cost breakdown endpoint"""
    
    response = await client.get(
        f"/api/v1/videos/{test_video.id}/cost",
        headers={"Authorization": f"Bearer {test_user.token}"}
    )
    
    assert response.status_code == 200
    data = response.json()
    
    # Verify cost breakdown
    assert "script_generation" in data
    assert "voice_synthesis" in data
    assert "video_processing" in data
    assert "total_cost" in data
    assert data["total_cost"] == sum(
        data[key] for key in ["script_generation", "voice_synthesis", "video_processing"]
    )
```

### 8.2 Integration Testing
```python
# tests/test_integration/test_video_pipeline.py
import pytest
from sqlalchemy.orm import Session

from app.services.video_processor import VideoProcessor
from app.services.ai_content import AIContentService
from app.services.youtube import YouTubeService

@pytest.mark.integration
@pytest.mark.asyncio
async def test_full_video_pipeline(
    db: Session,
    test_channel: Channel
):
    """Test complete video generation pipeline"""
    
    # Initialize services
    ai_service = AIContentService()
    video_processor = VideoProcessor()
    youtube_service = YouTubeService()
    
    # Generate script
    script_result = await ai_service.generate_video_script(
        topic="Test Video Topic",
        video_length=60  # 1 minute test
    )
    
    assert script_result["script"] is not None
    assert script_result["cost"] < 0.50
    
    # Process video (mocked for testing)
    video_result = await video_processor.create_video(
        script=script_result["script"],
        voice_settings={"voice_id": "test"},
        channel_id=test_channel.id
    )
    
    assert video_result["file_path"] is not None
    assert video_result["duration"] > 0
    
    # Verify total cost
    total_cost = script_result["cost"] + video_result["processing_cost"]
    assert total_cost < 1.0  # Under $1 target
```

---

## 9. Deployment & Operations

### 9.1 Local Development Setup
```bash
#!/bin/bash
# Local development setup script

# Create virtual environment
python3.11 -m venv venv
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
pip install -r requirements-dev.txt

# Setup environment
cp .env.example .env
# Edit .env with your configuration

# Initialize database
docker-compose up -d postgres redis
alembic upgrade head

# Run development server
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

### 9.2 Docker Configuration
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

# Run as non-root user
RUN useradd -m appuser && chown -R appuser:appuser /app
USER appuser

# Start application
CMD ["gunicorn", "app.main:app", "-w", "4", "-k", "uvicorn.workers.UvicornWorker", "--bind", "0.0.0.0:8000"]
```

---

## 10. MVP Delivery Checklist

### 10.1 Week 1-2: Foundation
- [ ] Set up project structure
- [ ] Configure database models
- [ ] Implement authentication
- [ ] Create base API endpoints
- [ ] Set up testing framework

### 10.2 Week 3-4: Core Features
- [ ] Channel management APIs
- [ ] Video generation endpoints
- [ ] YouTube integration
- [ ] Cost tracking system
- [ ] Basic analytics

### 10.3 Week 5-6: Integration
- [ ] N8N workflow integration
- [ ] External API connections
- [ ] File upload/storage
- [ ] Performance optimization
- [ ] Error handling

### 10.4 Week 7-8: Testing & Optimization
- [ ] Complete test coverage (70%+)
- [ ] Performance testing
- [ ] Security review
- [ ] API documentation
- [ ] Bug fixes

### 10.5 Week 9-10: Beta Preparation
- [ ] Load testing
- [ ] Monitoring setup
- [ ] Deployment scripts
- [ ] Admin tools
- [ ] User onboarding

### 10.6 Week 11-12: Launch Support
- [ ] Beta user support
- [ ] Performance monitoring
- [ ] Bug fixes
- [ ] Documentation updates
- [ ] Handover preparation

---

## Key Success Metrics

As an API Development Engineer, your success will be measured by:

1. **API Performance**: p95 latency <500ms
2. **Reliability**: 99.9% uptime for API endpoints
3. **Cost Efficiency**: <$1 per video generated
4. **Code Quality**: 70%+ test coverage
5. **Developer Experience**: Clear documentation, consistent APIs

Remember: You're building the foundation that will scale to support thousands of users managing hundreds of thousands of YouTube channels. Build it right the first time!

---

**Questions or Need Clarification?**
Contact your Backend Team Lead or refer to the YTEMPIRE technical documentation repository.