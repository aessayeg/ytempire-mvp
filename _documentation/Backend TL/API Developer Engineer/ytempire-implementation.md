# 7. IMPLEMENTATION GUIDES - YTEMPIRE

**Version 2.0 | January 2025**  
**Document Status: Consolidated & Standardized**  
**Last Updated: January 2025**

---

## 7.1 Development Environment Setup

### System Requirements

```yaml
Minimum Requirements:
  OS: Ubuntu 22.04 LTS / macOS 13+ / Windows 11 with WSL2
  CPU: 8 cores
  RAM: 16GB
  Storage: 50GB available
  Network: Stable internet connection

Recommended Setup:
  OS: Ubuntu 22.04 LTS
  CPU: 16 cores
  RAM: 32GB
  Storage: 100GB SSD
  GPU: NVIDIA with CUDA support (optional)
```

### Initial Setup Script

```bash
#!/bin/bash
# YTEMPIRE Development Environment Setup Script

echo "🚀 Setting up YTEMPIRE development environment..."

# Update system packages
sudo apt-get update && sudo apt-get upgrade -y

# Install system dependencies
sudo apt-get install -y \
    python3.11 \
    python3.11-dev \
    python3-pip \
    postgresql-14 \
    postgresql-client-14 \
    redis-server \
    nginx \
    git \
    curl \
    build-essential \
    libpq-dev

# Install Docker
curl -fsSL https://get.docker.com -o get-docker.sh
sudo sh get-docker.sh
sudo usermod -aG docker $USER

# Install Docker Compose
sudo curl -L "https://github.com/docker/compose/releases/download/v2.23.3/docker-compose-$(uname -s)-$(uname -m)" -o /usr/local/bin/docker-compose
sudo chmod +x /usr/local/bin/docker-compose

# Install Python tools
pip3 install --upgrade pip
pip3 install poetry virtualenv

# Clone repository
git clone https://github.com/ytempire/backend.git
cd backend

# Setup Python environment
python3.11 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
pip install -r requirements-dev.txt

# Setup environment variables
cp .env.example .env
echo "⚠️  Please edit .env file with your configuration"

# Initialize database
sudo -u postgres createdb ytempire
sudo -u postgres createuser ytempire_user

# Run database migrations
alembic upgrade head

# Install pre-commit hooks
pre-commit install

# Start services with Docker Compose
docker-compose up -d

echo "✅ Development environment setup complete!"
echo "📝 Next steps:"
echo "1. Edit .env file with your API keys"
echo "2. Run: source venv/bin/activate"
echo "3. Run: uvicorn app.main:app --reload"
```

### Environment Variables

```bash
# .env.example

# Application
APP_NAME=YTEMPIRE
APP_ENV=development
DEBUG=true
SECRET_KEY=your-secret-key-here-minimum-32-characters
API_V1_PREFIX=/v1

# Database
DATABASE_URL=postgresql://ytempire_user:password@localhost:5432/ytempire
DATABASE_POOL_SIZE=10
DATABASE_MAX_OVERFLOW=20

# Redis
REDIS_URL=redis://localhost:6379/0
REDIS_PASSWORD=

# Authentication
JWT_SECRET_KEY=your-jwt-secret-key-minimum-32-characters
JWT_ALGORITHM=HS256
ACCESS_TOKEN_EXPIRE_MINUTES=15
REFRESH_TOKEN_EXPIRE_DAYS=7

# YouTube API (15 accounts)
YOUTUBE_API_KEY_1=AIza...
YOUTUBE_CLIENT_ID_1=...
YOUTUBE_CLIENT_SECRET_1=...
# ... repeat for accounts 2-15

# OpenAI
OPENAI_API_KEY=sk-...
OPENAI_ORGANIZATION=org-...
OPENAI_MODEL=gpt-4-turbo-preview
OPENAI_MAX_TOKENS=4000

# ElevenLabs
ELEVENLABS_API_KEY=...
ELEVENLABS_VOICE_ID=...

# Stripe
STRIPE_API_KEY=sk_test_...
STRIPE_WEBHOOK_SECRET=whsec_...

# Storage
STORAGE_TYPE=local
STORAGE_PATH=/var/ytempire/storage
MAX_UPLOAD_SIZE=5368709120  # 5GB

# Monitoring
SENTRY_DSN=
PROMETHEUS_PORT=9090

# Rate Limiting
RATE_LIMIT_PER_MINUTE=60
RATE_LIMIT_PER_HOUR=1000

# Logging
LOG_LEVEL=INFO
LOG_FILE=/var/log/ytempire/app.log
```

### Docker Compose Configuration

```yaml
# docker-compose.yml
version: '3.8'

services:
  postgres:
    image: postgres:15-alpine
    environment:
      POSTGRES_DB: ytempire
      POSTGRES_USER: ytempire_user
      POSTGRES_PASSWORD: password
    ports:
      - "5432:5432"
    volumes:
      - postgres_data:/var/lib/postgresql/data
    healthcheck:
      test: ["CMD-SHELL", "pg_isready -U ytempire_user"]
      interval: 10s
      timeout: 5s
      retries: 5

  redis:
    image: redis:7-alpine
    command: redis-server --appendonly yes
    ports:
      - "6379:6379"
    volumes:
      - redis_data:/data
    healthcheck:
      test: ["CMD", "redis-cli", "ping"]
      interval: 10s
      timeout: 5s
      retries: 5

  n8n:
    image: n8nio/n8n:latest
    ports:
      - "5678:5678"
    environment:
      - N8N_BASIC_AUTH_ACTIVE=true
      - N8N_BASIC_AUTH_USER=admin
      - N8N_BASIC_AUTH_PASSWORD=password
    volumes:
      - n8n_data:/home/node/.n8n

  api:
    build: .
    ports:
      - "8000:8000"
    environment:
      - DATABASE_URL=postgresql://ytempire_user:password@postgres:5432/ytempire
      - REDIS_URL=redis://redis:6379/0
    depends_on:
      postgres:
        condition: service_healthy
      redis:
        condition: service_healthy
    volumes:
      - ./app:/app
      - ./storage:/storage
    command: uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload

volumes:
  postgres_data:
  redis_data:
  n8n_data:
```

---

## 7.2 Project Structure

### Directory Layout

```
ytempire-backend/
├── app/
│   ├── __init__.py
│   ├── main.py                 # FastAPI application entry point
│   ├── config.py               # Configuration management
│   ├── database.py             # Database connection and session
│   │
│   ├── api/
│   │   ├── __init__.py
│   │   ├── deps.py            # Common dependencies
│   │   └── v1/
│   │       ├── __init__.py
│   │       ├── auth.py        # Authentication endpoints
│   │       ├── users.py       # User management
│   │       ├── channels.py    # Channel operations
│   │       ├── videos.py      # Video generation
│   │       ├── analytics.py   # Analytics endpoints
│   │       └── admin.py       # Admin operations
│   │
│   ├── core/
│   │   ├── __init__.py
│   │   ├── config.py         # Core settings
│   │   ├── security.py       # Security utilities
│   │   ├── exceptions.py     # Custom exceptions
│   │   └── middleware.py     # Custom middleware
│   │
│   ├── crud/
│   │   ├── __init__.py
│   │   ├── base.py          # Base CRUD operations
│   │   ├── user.py          # User CRUD
│   │   ├── channel.py       # Channel CRUD
│   │   └── video.py         # Video CRUD
│   │
│   ├── models/
│   │   ├── __init__.py
│   │   ├── base.py          # Base model class
│   │   ├── user.py          # User model
│   │   ├── channel.py       # Channel model
│   │   ├── video.py         # Video model
│   │   └── analytics.py     # Analytics models
│   │
│   ├── schemas/
│   │   ├── __init__.py
│   │   ├── user.py          # User schemas
│   │   ├── channel.py       # Channel schemas
│   │   ├── video.py         # Video schemas
│   │   └── common.py        # Common schemas
│   │
│   ├── services/
│   │   ├── __init__.py
│   │   ├── auth.py          # Authentication service
│   │   ├── youtube.py       # YouTube API service
│   │   ├── openai.py        # OpenAI service
│   │   ├── elevenlabs.py    # ElevenLabs service
│   │   ├── cost.py          # Cost tracking service
│   │   └── queue.py         # Queue management
│   │
│   ├── tasks/
│   │   ├── __init__.py
│   │   ├── video.py         # Video generation tasks
│   │   ├── sync.py          # Sync tasks
│   │   └── cleanup.py       # Cleanup tasks
│   │
│   └── utils/
│       ├── __init__.py
│       ├── logger.py        # Logging configuration
│       ├── cache.py         # Cache utilities
│       └── validators.py    # Custom validators
│
├── migrations/                 # Alembic migrations
│   ├── alembic.ini
│   ├── env.py
│   └── versions/
│
├── tests/                      # Test suite
│   ├── __init__.py
│   ├── conftest.py           # Pytest configuration
│   ├── test_api/
│   ├── test_crud/
│   ├── test_services/
│   └── test_utils/
│
├── scripts/                    # Utility scripts
│   ├── init_db.py
│   ├── seed_data.py
│   └── backup_db.sh
│
├── docs/                       # Documentation
│   ├── api/
│   ├── deployment/
│   └── development/
│
├── .github/                    # GitHub configuration
│   └── workflows/
│       ├── test.yml
│       └── deploy.yml
│
├── Dockerfile                  # Container definition
├── docker-compose.yml         # Local development stack
├── requirements.txt           # Python dependencies
├── requirements-dev.txt       # Development dependencies
├── .env.example              # Environment template
├── .gitignore                # Git ignore rules
├── .pre-commit-config.yaml   # Pre-commit hooks
├── pyproject.toml            # Python project config
└── README.md                 # Project documentation
```

---

## 7.3 Core Features Implementation

### Authentication System

```python
# app/services/auth.py
from datetime import datetime, timedelta
from typing import Optional
from jose import JWTError, jwt
from passlib.context import CryptContext
from sqlalchemy.orm import Session
from app.core.config import settings
from app.models.user import User
from app.schemas.auth import TokenData

class AuthService:
    def __init__(self):
        self.pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")
        
    def create_access_token(self, data: dict) -> str:
        to_encode = data.copy()
        expire = datetime.utcnow() + timedelta(minutes=settings.ACCESS_TOKEN_EXPIRE_MINUTES)
        to_encode.update({"exp": expire, "type": "access"})
        return jwt.encode(to_encode, settings.JWT_SECRET_KEY, algorithm=settings.JWT_ALGORITHM)
    
    def create_refresh_token(self, data: dict) -> str:
        to_encode = data.copy()
        expire = datetime.utcnow() + timedelta(days=settings.REFRESH_TOKEN_EXPIRE_DAYS)
        to_encode.update({"exp": expire, "type": "refresh"})
        return jwt.encode(to_encode, settings.JWT_SECRET_KEY, algorithm=settings.JWT_ALGORITHM)
    
    def verify_token(self, token: str, token_type: str = "access") -> Optional[TokenData]:
        try:
            payload = jwt.decode(token, settings.JWT_SECRET_KEY, algorithms=[settings.JWT_ALGORITHM])
            if payload.get("type") != token_type:
                return None
            return TokenData(**payload)
        except JWTError:
            return None
    
    def authenticate_user(self, db: Session, email: str, password: str) -> Optional[User]:
        user = db.query(User).filter(User.email == email).first()
        if not user or not self.pwd_context.verify(password, user.password_hash):
            return None
        return user
    
    def get_password_hash(self, password: str) -> str:
        return self.pwd_context.hash(password)

auth_service = AuthService()
```

### Video Generation Pipeline

```python
# app/services/video.py
import asyncio
from typing import Dict, Any
from uuid import UUID
from datetime import datetime
from sqlalchemy.orm import Session
from app.models.video import Video, VideoStatus
from app.services.openai import openai_service
from app.services.elevenlabs import elevenlabs_service
from app.services.youtube import youtube_service
from app.services.cost import cost_service
import logging

logger = logging.getLogger(__name__)

class VideoGenerationService:
    async def generate_video(self, db: Session, video_id: UUID) -> Dict[str, Any]:
        """Main video generation pipeline"""
        video = db.query(Video).filter(Video.id == video_id).first()
        if not video:
            raise ValueError(f"Video {video_id} not found")
        
        try:
            # Update status to processing
            video.status = VideoStatus.PROCESSING
            video.processing_started_at = datetime.utcnow()
            db.commit()
            
            # Step 1: Generate script
            logger.info(f"Generating script for video {video_id}")
            script = await self._generate_script(video)
            cost_service.track_cost(db, video_id, "openai", script['cost'])
            
            # Step 2: Generate voice
            logger.info(f"Generating voice for video {video_id}")
            audio = await self._generate_voice(script['text'])
            cost_service.track_cost(db, video_id, "elevenlabs", audio['cost'])
            
            # Step 3: Generate thumbnail
            logger.info(f"Generating thumbnail for video {video_id}")
            thumbnail = await self._generate_thumbnail(video.topic)
            cost_service.track_cost(db, video_id, "stability", thumbnail['cost'])
            
            # Step 4: Combine into video
            logger.info(f"Creating video for {video_id}")
            video_file = await self._create_video(audio['path'], thumbnail['path'])
            
            # Step 5: Calculate quality score
            quality_score = await self._calculate_quality(script, audio, thumbnail)
            
            # Update video record
            video.status = VideoStatus.COMPLETED
            video.script_path = script['path']
            video.audio_path = audio['path']
            video.thumbnail_path = thumbnail['path']
            video.video_path = video_file['path']
            video.quality_score = quality_score
            video.total_cost = cost_service.get_total_cost(db, video_id)
            video.processing_completed_at = datetime.utcnow()
            video.duration_seconds = audio['duration']
            db.commit()
            
            logger.info(f"Video {video_id} generated successfully")
            return {
                "success": True,
                "video_id": str(video_id),
                "cost": video.total_cost,
                "quality_score": quality_score
            }
            
        except Exception as e:
            logger.error(f"Error generating video {video_id}: {str(e)}")
            video.status = VideoStatus.FAILED
            video.error_message = str(e)
            db.commit()
            raise
    
    async def _generate_script(self, video: Video) -> Dict[str, Any]:
        """Generate video script using OpenAI"""
        prompt = f"""
        Create a YouTube video script about: {video.topic}
        Style: {video.style}
        Target duration: {video.duration_target} seconds
        
        Requirements:
        - Engaging hook in first 5 seconds
        - Clear structure with introduction, main points, and conclusion
        - Natural conversational tone
        - Include calls-to-action
        """
        
        response = await openai_service.generate_script(prompt)
        
        # Save script to file
        script_path = f"/storage/scripts/{video.id}.txt"
        with open(script_path, 'w') as f:
            f.write(response['text'])
        
        return {
            "text": response['text'],
            "path": script_path,
            "cost": response['cost']
        }
    
    async def _generate_voice(self, script: str) -> Dict[str, Any]:
        """Generate voice using ElevenLabs"""
        response = await elevenlabs_service.text_to_speech(
            text=script,
            voice_id=settings.ELEVENLABS_VOICE_ID
        )
        
        return {
            "path": response['audio_path'],
            "duration": response['duration'],
            "cost": response['cost']
        }
    
    async def _generate_thumbnail(self, topic: str) -> Dict[str, Any]:
        """Generate thumbnail using AI"""
        # Implementation for thumbnail generation
        # This would use Stability AI or similar service
        return {
            "path": f"/storage/thumbnails/{UUID()}.png",
            "cost": 0.05
        }
    
    async def _create_video(self, audio_path: str, thumbnail_path: str) -> Dict[str, Any]:
        """Combine audio and thumbnail into video"""
        # Implementation using ffmpeg or similar
        # This would create a video with the thumbnail as background
        # and the audio track
        return {
            "path": f"/storage/videos/{UUID()}.mp4"
        }
    
    async def _calculate_quality(self, script: Dict, audio: Dict, thumbnail: Dict) -> float:
        """Calculate quality score for the video"""
        # Implementation for quality scoring
        # Could use AI evaluation or rule-based scoring
        return 0.85

video_service = VideoGenerationService()
```

### Cost Tracking System

```python
# app/services/cost.py
from decimal import Decimal
from typing import Dict, List
from uuid import UUID
from sqlalchemy.orm import Session
from sqlalchemy import func
from app.models.video import VideoCost
from datetime import datetime, timedelta

class CostTrackingService:
    def __init__(self):
        self.cost_limits = {
            "daily_total": Decimal("150.00"),
            "per_video": Decimal("3.00"),
            "alert_threshold": Decimal("100.00")
        }
    
    def track_cost(self, db: Session, video_id: UUID, service: str, amount: Decimal, 
                   operation: str = None, metadata: Dict = None) -> VideoCost:
        """Track cost for a specific service"""
        cost = VideoCost(
            video_id=video_id,
            service=service,
            operation=operation,
            amount=amount,
            metadata=metadata or {}
        )
        db.add(cost)
        db.commit()
        
        # Check limits
        self._check_limits(db, video_id)
        
        return cost
    
    def get_total_cost(self, db: Session, video_id: UUID) -> Decimal:
        """Get total cost for a video"""
        total = db.query(func.sum(VideoCost.amount)).filter(
            VideoCost.video_id == video_id
        ).scalar()
        return total or Decimal("0.00")
    
    def get_daily_spend(self, db: Session, date: datetime = None) -> Decimal:
        """Get total spend for a specific day"""
        if date is None:
            date = datetime.utcnow().date()
        
        start = datetime.combine(date, datetime.min.time())
        end = start + timedelta(days=1)
        
        total = db.query(func.sum(VideoCost.amount)).filter(
            VideoCost.created_at >= start,
            VideoCost.created_at < end
        ).scalar()
        
        return total or Decimal("0.00")
    
    def _check_limits(self, db: Session, video_id: UUID):
        """Check if costs exceed limits"""
        video_total = self.get_total_cost(db, video_id)
        if video_total > self.cost_limits["per_video"]:
            raise ValueError(f"Video cost ${video_total} exceeds limit ${self.cost_limits['per_video']}")
        
        daily_total = self.get_daily_spend(db)
        if daily_total > self.cost_limits["daily_total"]:
            raise ValueError(f"Daily spend ${daily_total} exceeds limit ${self.cost_limits['daily_total']}")
    
    def get_cost_breakdown(self, db: Session, video_id: UUID) -> Dict[str, Decimal]:
        """Get cost breakdown by service"""
        costs = db.query(
            VideoCost.service,
            func.sum(VideoCost.amount).label('total')
        ).filter(
            VideoCost.video_id == video_id
        ).group_by(VideoCost.service).all()
        
        return {cost.service: cost.total for cost in costs}

cost_service = CostTrackingService()
```

---

## 7.4 Integration Patterns

### YouTube API Integration

```python
# app/services/youtube.py
from google.auth.transport.requests import Request
from google.oauth2.credentials import Credentials
from googleapiclient.discovery import build
from googleapiclient.errors import HttpError
from googleapiclient.http import MediaFileUpload
import json
import random
from typing import Dict, List, Optional
from datetime import datetime, timedelta

class YouTubeService:
    def __init__(self):
        self.accounts = self._load_accounts()
        self.current_account_index = 0
        self.quota_tracker = {}
        
    def _load_accounts(self) -> List[Dict]:
        """Load all 15 YouTube accounts (10 active + 5 reserve)"""
        accounts = []
        for i in range(1, 16):
            creds_file = f"/secrets/youtube_account_{i}.json"
            with open(creds_file, 'r') as f:
                creds_data = json.load(f)
                accounts.append({
                    'index': i,
                    'credentials': Credentials.from_authorized_user_info(creds_data),
                    'is_reserve': i > 10,  # Accounts 11-15 are reserve
                    'quota_used': 0,
                    'uploads_today': 0,
                    'last_reset': datetime.utcnow().date()
                })
        return accounts
    
    def _get_available_account(self) -> Dict:
        """Get an available account with quota remaining"""
        # Reset daily quotas if needed
        self._reset_daily_quotas()
        
        # Try active accounts first
        active_accounts = [a for a in self.accounts if not a['is_reserve']]
        for account in active_accounts:
            if account['quota_used'] < 9000 and account['uploads_today'] < 5:
                return account
        
        # Fall back to reserve accounts
        reserve_accounts = [a for a in self.accounts if a['is_reserve']]
        for account in reserve_accounts:
            if account['quota_used'] < 9000 and account['uploads_today'] < 5:
                return account
        
        raise Exception("No YouTube accounts available with remaining quota")
    
    def _reset_daily_quotas(self):
        """Reset quotas at midnight Pacific Time"""
        current_date = datetime.utcnow().date()
        for account in self.accounts:
            if account['last_reset'] < current_date:
                account['quota_used'] = 0
                account['uploads_today'] = 0
                account['last_reset'] = current_date
    
    async def upload_video(self, video_path: str, title: str, description: str, 
                          tags: List[str], category_id: str = "22") -> Dict:
        """Upload video to YouTube with quota management"""
        account = self._get_available_account()
        
        # Build YouTube API client
        youtube = build('youtube', 'v3', credentials=account['credentials'])
        
        body = {
            'snippet': {
                'title': title,
                'description': description,
                'tags': tags,
                'categoryId': category_id
            },
            'status': {
                'privacyStatus': 'public',
                'selfDeclaredMadeForKids': False
            }
        }
        
        # Upload with resumable upload for large files
        media = MediaFileUpload(
            video_path,
            mimetype='video/mp4',
            resumable=True,
            chunksize=50 * 1024 * 1024  # 50MB chunks
        )
        
        try:
            request = youtube.videos().insert(
                part=','.join(body.keys()),
                body=body,
                media_body=media
            )
            
            response = None
            while response is None:
                status, response = request.next_chunk()
                if status:
                    print(f"Upload progress: {int(status.progress() * 100)}%")
            
            # Update quota tracking
            account['quota_used'] += 1600  # Upload costs ~1600 quota units
            account['uploads_today'] += 1
            
            return {
                'youtube_id': response['id'],
                'youtube_url': f"https://youtube.com/watch?v={response['id']}",
                'account_used': account['index']
            }
            
        except HttpError as e:
            if e.resp.status == 403 and 'quotaExceeded' in str(e):
                # Mark account as exhausted and retry with different account
                account['quota_used'] = 10000
                return await self.upload_video(video_path, title, description, tags, category_id)
            raise

youtube_service = YouTubeService()
```

### OpenAI Integration

```python
# app/services/openai.py
import openai
from typing import Dict, Optional
from decimal import Decimal
import tiktoken
from app.core.config import settings

class OpenAIService:
    def __init__(self):
        openai.api_key = settings.OPENAI_API_KEY
        self.encoding = tiktoken.encoding_for_model("gpt-4")
        self.cost_per_1k_tokens = {
            "gpt-4-turbo-preview": {"input": 0.01, "output": 0.03},
            "gpt-3.5-turbo": {"input": 0.0005, "output": 0.0015}
        }
    
    async def generate_script(self, prompt: str, model: str = None) -> Dict:
        """Generate video script with cost tracking"""
        if model is None:
            model = settings.OPENAI_MODEL
        
        # Count input tokens
        input_tokens = len(self.encoding.encode(prompt))
        
        try:
            response = await openai.ChatCompletion.acreate(
                model=model,
                messages=[
                    {"role": "system", "content": "You are a professional YouTube scriptwriter."},
                    {"role": "user", "content": prompt}
                ],
                max_tokens=settings.OPENAI_MAX_TOKENS,
                temperature=0.7,
                top_p=0.9
            )
            
            output_text = response.choices[0].message.content
            output_tokens = response.usage.completion_tokens
            total_tokens = response.usage.total_tokens
            
            # Calculate cost
            cost = self._calculate_cost(model, input_tokens, output_tokens)
            
            return {
                "text": output_text,
                "tokens": total_tokens,
                "cost": cost,
                "model": model
            }
            
        except openai.error.RateLimitError:
            # Fallback to GPT-3.5 if rate limited
            if model != "gpt-3.5-turbo":
                return await self.generate_script(prompt, "gpt-3.5-turbo")
            raise
    
    def _calculate_cost(self, model: str, input_tokens: int, output_tokens: int) -> Decimal:
        """Calculate cost for OpenAI API usage"""
        rates = self.cost_per_1k_tokens.get(model, self.cost_per_1k_tokens["gpt-4-turbo-preview"])
        
        input_cost = Decimal(str(input_tokens / 1000 * rates["input"]))
        output_cost = Decimal(str(output_tokens / 1000 * rates["output"]))
        
        return input_cost + output_cost

openai_service = OpenAIService()
```

### Webhook Integration

```python
# app/api/v1/webhooks.py
from fastapi import APIRouter, HTTPException, Header, Request
from typing import Optional
import hmac
import hashlib
import stripe
from app.core.config import settings

router = APIRouter(prefix="/webhooks", tags=["webhooks"])

@router.post("/stripe")
async def stripe_webhook(
    request: Request,
    stripe_signature: Optional[str] = Header(None)
):
    """Handle Stripe webhook events"""
    payload = await request.body()
    
    # Verify webhook signature
    try:
        sig_header = stripe_signature
        event = stripe.Webhook.construct_event(
            payload, sig_header, settings.STRIPE_WEBHOOK_SECRET
        )
    except ValueError:
        raise HTTPException(status_code=400, detail="Invalid payload")
    except stripe.error.SignatureVerificationError:
        raise HTTPException(status_code=400, detail="Invalid signature")
    
    # Handle different event types
    if event['type'] == 'payment_intent.succeeded':
        # Handle successful payment
        pass
    elif event['type'] == 'customer.subscription.created':
        # Handle new subscription
        pass
    elif event['type'] == 'customer.subscription.deleted':
        # Handle cancelled subscription
        pass
    
    return {"status": "success"}

@router.post("/n8n/{workflow_id}")
async def n8n_webhook(workflow_id: str, request: Request):
    """Handle N8N workflow webhooks"""
    data = await request.json()
    
    # Process based on workflow
    if workflow_id == "video_completed":
        # Handle video completion
        video_id = data.get("video_id")
        # Update video status, trigger next steps
        pass
    elif workflow_id == "quality_check":
        # Handle quality check results
        pass
    
    return {"status": "received", "workflow_id": workflow_id}
```

---

## 7.5 Testing Strategy

### Test Configuration

```python
# tests/conftest.py
import pytest
from typing import Generator
from fastapi.testclient import TestClient
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker, Session
from app.main import app
from app.database import Base, get_db
from app.core.config import settings

# Use in-memory SQLite for tests
SQLALCHEMY_DATABASE_URL = "sqlite:///./test.db"

engine = create_engine(
    SQLALCHEMY_DATABASE_URL, connect_args={"check_same_thread": False}
)
TestingSessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

@pytest.fixture(scope="session")
def db() -> Generator:
    Base.metadata.create_all(bind=engine)
    yield TestingSessionLocal()
    Base.metadata.drop_all(bind=engine)

@pytest.fixture(scope="module")
def client() -> Generator:
    with TestClient(app) as c:
        yield c

@pytest.fixture
def test_user(db: Session):
    """Create a test user"""
    from app.models.user import User
    from app.services.auth import auth_service
    
    user = User(
        email="test@example.com",
        username="testuser",
        password_hash=auth_service.get_password_hash("testpass123"),
        tier="growth"
    )
    db.add(user)
    db.commit()
    db.refresh(user)
    return user

@pytest.fixture
def auth_headers(test_user):
    """Generate auth headers for test user"""
    from app.services.auth import auth_service
    
    access_token = auth_service.create_access_token(
        data={"sub": str(test_user.id), "email": test_user.email}
    )
    return {"Authorization": f"Bearer {access_token}"}
```

### API Tests

```python
# tests/test_api/test_videos.py
import pytest
from fastapi.testclient import TestClient
from sqlalchemy.orm import Session

def test_create_video(client: TestClient, auth_headers: dict, db: Session):
    """Test video creation endpoint"""
    response = client.post(
        "/v1/videos/generate",
        headers=auth_headers,
        json={
            "channel_id": "ch_1234567890abcdef",
            "topic": "Python Tutorial",
            "style": "educational",
            "duration_target": 600
        }
    )
    
    assert response.status_code == 202
    data = response.json()
    assert data["success"] is True
    assert "video_id" in data["data"]

def test_get_video(client: TestClient, auth_headers: dict, test_video):
    """Test getting video details"""
    response = client.get(
        f"/v1/videos/{test_video.id}",
        headers=auth_headers
    )
    
    assert response.status_code == 200
    data = response.json()
    assert data["data"]["id"] == str(test_video.id)

def test_list_videos(client: TestClient, auth_headers: dict):
    """Test listing videos"""
    response = client.get(
        "/v1/videos",
        headers=auth_headers,
        params={"page": 1, "per_page": 10}
    )
    
    assert response.status_code == 200
    data = response.json()
    assert "data" in data
    assert "pagination" in data

def test_video_cost_tracking(client: TestClient, auth_headers: dict, test_video):
    """Test cost tracking for video"""
    response = client.get(
        f"/v1/videos/{test_video.id}/cost",
        headers=auth_headers
    )
    
    assert response.status_code == 200
    data = response.json()
    assert "total_cost" in data["data"]
    assert "breakdown" in data["data"]
```

### Service Tests

```python
# tests/test_services/test_cost_service.py
import pytest
from decimal import Decimal
from uuid import uuid4
from app.services.cost import cost_service
from sqlalchemy.orm import Session

def test_track_cost(db: Session):
    """Test cost tracking"""
    video_id = uuid4()
    
    cost = cost_service.track_cost(
        db=db,
        video_id=video_id,
        service="openai",
        amount=Decimal("0.45"),
        operation="script_generation"
    )
    
    assert cost.video_id == video_id
    assert cost.service == "openai"
    assert cost.amount == Decimal("0.45")

def test_cost_limits(db: Session):
    """Test cost limit enforcement"""
    video_id = uuid4()
    
    # Track costs up to limit
    for i in range(10):
        cost_service.track_cost(
            db=db,
            video_id=video_id,
            service="test",
            amount=Decimal("0.29")
        )
    
    # This should exceed the $3 limit
    with pytest.raises(ValueError, match="exceeds limit"):
        cost_service.track_cost(
            db=db,
            video_id=video_id,
            service="test",
            amount=Decimal("0.20")
        )

def test_daily_spend_limit(db: Session):
    """Test daily spend limit"""
    # Create multiple videos with costs
    for i in range(50):
        video_id = uuid4()
        cost_service.track_cost(
            db=db,
            video_id=video_id,
            service="test",
            amount=Decimal("2.99")
        )
    
    # This should exceed daily limit
    with pytest.raises(ValueError, match="Daily spend"):
        cost_service.track_cost(
            db=db,
            video_id=uuid4(),
            service="test",
            amount=Decimal("2.00")
        )
```

### Performance Tests

```python
# tests/test_performance/test_api_performance.py
import pytest
import asyncio
import time
from fastapi.testclient import TestClient
from concurrent.futures import ThreadPoolExecutor

def test_api_response_time(client: TestClient, auth_headers: dict):
    """Test API response time meets SLA"""
    response_times = []
    
    for _ in range(100):
        start = time.time()
        response = client.get("/v1/channels", headers=auth_headers)
        end = time.time()
        
        assert response.status_code == 200
        response_times.append((end - start) * 1000)  # Convert to ms
    
    # Check p95 < 500ms
    response_times.sort()
    p95 = response_times[int(len(response_times) * 0.95)]
    assert p95 < 500, f"p95 response time {p95}ms exceeds 500ms SLA"

def test_concurrent_requests(client: TestClient, auth_headers: dict):
    """Test handling concurrent requests"""
    def make_request():
        return client.get("/v1/videos", headers=auth_headers)
    
    with ThreadPoolExecutor(max_workers=50) as executor:
        futures = [executor.submit(make_request) for _ in range(100)]
        results = [f.result() for f in futures]
    
    # All requests should succeed
    for response in results:
        assert response.status_code == 200

def test_database_query_performance(db: Session):
    """Test database query performance"""
    from app.crud.video import video_crud
    
    start = time.time()
    videos = video_crud.get_multi(db, skip=0, limit=100)
    end = time.time()
    
    query_time = (end - start) * 1000
    assert query_time < 100, f"Query took {query_time}ms, exceeds 100ms limit"
```

---

## Document Control

- **Version**: 2.0
- **Last Updated**: January 2025
- **Next Review**: February 2025
- **Owner**: API Development Engineer
- **Approved By**: Backend Team Lead

---

## Navigation

- [← Previous: Database Design](./6-database-design.md)
- [→ Next: External Integrations](./8-external-integrations.md)