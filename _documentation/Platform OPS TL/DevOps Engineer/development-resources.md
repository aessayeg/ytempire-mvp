# YTEMPIRE Documentation - Development Resources

## 6.1 API Documentation

### REST API Specification

#### Base URL
```
Production: https://api.ytempire.com/api/v1
Staging: https://staging-api.ytempire.com/api/v1
Development: http://localhost:8080/api/v1
```

#### Authentication
All API requests require JWT authentication except for `/auth/register` and `/auth/login`.

```http
Authorization: Bearer <jwt_token>
```

### Authentication Endpoints

#### Register User
```http
POST /auth/register

Request Body:
{
  "email": "user@example.com",
  "password": "SecurePassword123!",
  "subscription_plan": "starter"
}

Response (201 Created):
{
  "user_id": "550e8400-e29b-41d4-a716-446655440000",
  "email": "user@example.com",
  "subscription_plan": "starter",
  "created_at": "2025-01-20T10:00:00Z"
}

Error Response (400 Bad Request):
{
  "error": "Email already registered"
}
```

#### Login
```http
POST /auth/login

Request Body:
{
  "email": "user@example.com",
  "password": "SecurePassword123!"
}

Response (200 OK):
{
  "access_token": "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9...",
  "refresh_token": "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9...",
  "token_type": "Bearer",
  "expires_in": 3600
}

Error Response (401 Unauthorized):
{
  "error": "Invalid credentials"
}
```

#### Refresh Token
```http
POST /auth/refresh

Request Body:
{
  "refresh_token": "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9..."
}

Response (200 OK):
{
  "access_token": "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9...",
  "token_type": "Bearer",
  "expires_in": 3600
}
```

### Channel Management Endpoints

#### List Channels
```http
GET /channels

Query Parameters:
- page (integer): Page number (default: 1)
- limit (integer): Items per page (default: 10, max: 50)
- status (string): Filter by status (active, paused, archived)

Response (200 OK):
{
  "channels": [
    {
      "id": "channel-uuid",
      "name": "Tech Reviews Channel",
      "youtube_channel_id": "UC123456",
      "niche": "technology",
      "status": "active",
      "subscriber_count": 1523,
      "video_count": 45,
      "monthly_revenue": 234.56,
      "created_at": "2025-01-15T10:00:00Z"
    }
  ],
  "pagination": {
    "page": 1,
    "limit": 10,
    "total": 5,
    "pages": 1
  }
}
```

#### Create Channel
```http
POST /channels

Request Body:
{
  "name": "New Channel Name",
  "niche": "technology",
  "youtube_credentials": {
    "client_id": "oauth_client_id",
    "client_secret": "oauth_client_secret",
    "refresh_token": "oauth_refresh_token"
  },
  "settings": {
    "upload_schedule": "daily",
    "videos_per_day": 2,
    "auto_publish": true
  }
}

Response (201 Created):
{
  "id": "channel-uuid",
  "name": "New Channel Name",
  "niche": "technology",
  "status": "pending_setup",
  "created_at": "2025-01-20T10:00:00Z"
}

Error Response (400 Bad Request):
{
  "error": "Maximum channel limit reached for your subscription"
}
```

#### Update Channel
```http
PUT /channels/{channel_id}

Request Body:
{
  "name": "Updated Channel Name",
  "status": "paused",
  "settings": {
    "upload_schedule": "weekly",
    "videos_per_week": 3
  }
}

Response (200 OK):
{
  "id": "channel-uuid",
  "name": "Updated Channel Name",
  "status": "paused",
  "updated_at": "2025-01-20T10:30:00Z"
}
```

### Video Generation Endpoints

#### Generate Video
```http
POST /videos/generate

Request Body:
{
  "channel_id": "channel-uuid",
  "topic": "Latest iPhone Review",
  "style": "review",
  "length": "medium",
  "voice": "professional_male",
  "thumbnail_style": "modern",
  "advanced_options": {
    "use_stock_footage": true,
    "add_captions": true,
    "music_track": "upbeat_tech"
  }
}

Response (202 Accepted):
{
  "video_id": "video-uuid",
  "status": "queued",
  "estimated_completion": "2025-01-20T10:10:00Z",
  "queue_position": 3,
  "estimated_cost": 2.45
}
```

#### Get Video Status
```http
GET /videos/{video_id}/status

Response (200 OK):
{
  "video_id": "video-uuid",
  "status": "processing",
  "progress": 65,
  "current_stage": "voice_synthesis",
  "stages": {
    "script_generation": "completed",
    "voice_synthesis": "in_progress",
    "video_assembly": "pending",
    "thumbnail_generation": "pending",
    "upload": "pending"
  },
  "started_at": "2025-01-20T10:05:00Z",
  "estimated_completion": "2025-01-20T10:12:00Z"
}
```

#### List Videos
```http
GET /videos

Query Parameters:
- channel_id (uuid): Filter by channel
- status (string): pending, processing, completed, failed
- date_from (date): Start date filter
- date_to (date): End date filter
- page (integer): Page number
- limit (integer): Items per page

Response (200 OK):
{
  "videos": [
    {
      "id": "video-uuid",
      "channel_id": "channel-uuid",
      "title": "iPhone 15 Pro Max Review",
      "youtube_video_id": "dQw4w9WgXcQ",
      "status": "completed",
      "views": 1523,
      "likes": 89,
      "revenue": 12.34,
      "cost": 2.45,
      "created_at": "2025-01-20T09:00:00Z",
      "published_at": "2025-01-20T10:00:00Z"
    }
  ],
  "pagination": {
    "page": 1,
    "limit": 20,
    "total": 45,
    "pages": 3
  }
}
```

### Analytics Endpoints

#### Dashboard Metrics
```http
GET /analytics/dashboard

Query Parameters:
- period (string): today, week, month, year
- channel_id (uuid): Optional channel filter

Response (200 OK):
{
  "summary": {
    "total_channels": 5,
    "total_videos": 234,
    "total_views": 45678,
    "total_revenue": 1234.56,
    "total_cost": 234.56,
    "profit": 1000.00,
    "profit_margin": 81.0
  },
  "trends": {
    "views": [
      {"date": "2025-01-14", "value": 5234},
      {"date": "2025-01-15", "value": 6123},
      {"date": "2025-01-16", "value": 7234}
    ],
    "revenue": [
      {"date": "2025-01-14", "value": 123.45},
      {"date": "2025-01-15", "value": 145.67},
      {"date": "2025-01-16", "value": 167.89}
    ]
  },
  "top_performing": {
    "videos": [
      {
        "id": "video-uuid",
        "title": "Top Video Title",
        "views": 12345,
        "revenue": 234.56
      }
    ],
    "channels": [
      {
        "id": "channel-uuid",
        "name": "Top Channel",
        "total_views": 23456,
        "total_revenue": 456.78
      }
    ]
  }
}
```

### WebSocket Events

#### Connection
```javascript
const ws = new WebSocket('wss://api.ytempire.com/ws');

ws.onopen = () => {
  // Authenticate
  ws.send(JSON.stringify({
    type: 'auth',
    token: 'jwt_token_here'
  }));
  
  // Subscribe to events
  ws.send(JSON.stringify({
    type: 'subscribe',
    events: ['video_progress', 'channel_updates', 'analytics']
  }));
};
```

#### Event Types
```javascript
// Video Progress Event
{
  "event": "video_progress",
  "data": {
    "video_id": "video-uuid",
    "status": "processing",
    "progress": 75,
    "stage": "video_assembly"
  }
}

// Channel Update Event
{
  "event": "channel_update",
  "data": {
    "channel_id": "channel-uuid",
    "type": "video_published",
    "video_id": "video-uuid",
    "youtube_url": "https://youtube.com/watch?v=..."
  }
}

// Analytics Update Event
{
  "event": "analytics_update",
  "data": {
    "channel_id": "channel-uuid",
    "metrics": {
      "views": 1234,
      "revenue": 12.34,
      "subscribers": 5678
    }
  }
}
```

### Error Responses

#### Standard Error Format
```json
{
  "error": {
    "code": "RESOURCE_NOT_FOUND",
    "message": "The requested resource was not found",
    "details": {
      "resource": "channel",
      "id": "invalid-uuid"
    }
  },
  "request_id": "req_123456789"
}
```

#### Common Error Codes
| Code | HTTP Status | Description |
|------|-------------|-------------|
| `INVALID_REQUEST` | 400 | Request validation failed |
| `UNAUTHORIZED` | 401 | Authentication required |
| `FORBIDDEN` | 403 | Insufficient permissions |
| `RESOURCE_NOT_FOUND` | 404 | Resource doesn't exist |
| `CONFLICT` | 409 | Resource conflict |
| `RATE_LIMITED` | 429 | Too many requests |
| `INTERNAL_ERROR` | 500 | Server error |

## 6.2 Development Environment

### Local Setup Guide

#### Prerequisites
- Ubuntu 22.04 or macOS 13+
- Docker 24.0+
- Docker Compose 2.20+
- Python 3.11+
- Node.js 18+
- Git 2.40+

#### Initial Setup
```bash
# 1. Clone repository
git clone https://github.com/ytempire/ytempire.git
cd ytempire

# 2. Copy environment variables
cp .env.example .env
# Edit .env with your configuration

# 3. Install dependencies
make install

# 4. Start services
docker-compose up -d

# 5. Run migrations
make migrate

# 6. Seed test data
make seed

# 7. Start development servers
make dev
```

### Environment Variables

#### Required Variables
```bash
# Database
DATABASE_URL=postgresql://ytempire:password@localhost:5432/ytempire
REDIS_URL=redis://localhost:6379/0

# Security
JWT_SECRET=your-secret-key-minimum-32-characters
JWT_EXPIRY=3600

# External APIs
OPENAI_API_KEY=sk-...
ELEVENLABS_API_KEY=...
YOUTUBE_CLIENT_ID=...
YOUTUBE_CLIENT_SECRET=...
STRIPE_SECRET_KEY=sk_test_...

# Storage
STORAGE_PATH=/opt/ytempire/storage
BACKUP_PATH=/opt/ytempire/backup

# Monitoring
PROMETHEUS_ENABLED=true
GRAFANA_ADMIN_PASSWORD=admin

# Development
DEBUG=true
LOG_LEVEL=debug
```

### Docker Development

#### Local Docker Compose
```yaml
# docker-compose.dev.yml
version: '3.9'

services:
  api:
    build:
      context: ./backend
      dockerfile: Dockerfile.dev
    volumes:
      - ./backend:/app
      - /app/__pycache__
    environment:
      - FLASK_ENV=development
      - FLASK_DEBUG=1
    ports:
      - "8080:8080"
      - "5555:5555"  # Debugger port

  frontend:
    build:
      context: ./frontend
      dockerfile: Dockerfile.dev
    volumes:
      - ./frontend:/app
      - /app/node_modules
    environment:
      - NODE_ENV=development
    ports:
      - "3000:3000"
    command: npm run dev
```

### Development Tools

#### Makefile Commands
```makefile
# Makefile
.PHONY: help install dev test lint format clean

help:
	@echo "Available commands:"
	@echo "  install    Install all dependencies"
	@echo "  dev        Start development servers"
	@echo "  test       Run all tests"
	@echo "  lint       Run linters"
	@echo "  format     Format code"
	@echo "  clean      Clean up containers and cache"

install:
	cd backend && pip install -r requirements.txt -r requirements-dev.txt
	cd frontend && npm install
	docker-compose pull

dev:
	docker-compose -f docker-compose.dev.yml up

test:
	cd backend && pytest tests/ --cov=app
	cd frontend && npm test

lint:
	cd backend && pylint app/ && mypy app/
	cd frontend && npm run lint

format:
	cd backend && black app/ tests/
	cd frontend && npm run format

clean:
	docker-compose down -v
	find . -type d -name "__pycache__" -exec rm -rf {} +
	find . -type f -name "*.pyc" -delete
	rm -rf frontend/node_modules
	rm -rf .pytest_cache
```

## 6.3 Testing Strategies

### Unit Testing

#### Backend Testing (Python/Pytest)
```python
# tests/test_api.py
import pytest
from fastapi.testclient import TestClient
from app.main import app

client = TestClient(app)

class TestAuthentication:
    def test_register_user(self):
        response = client.post("/api/v1/auth/register", json={
            "email": "test@example.com",
            "password": "TestPassword123!"
        })
        assert response.status_code == 201
        assert "user_id" in response.json()
    
    def test_login_valid_credentials(self):
        response = client.post("/api/v1/auth/login", json={
            "email": "test@example.com",
            "password": "TestPassword123!"
        })
        assert response.status_code == 200
        assert "access_token" in response.json()
    
    def test_login_invalid_credentials(self):
        response = client.post("/api/v1/auth/login", json={
            "email": "test@example.com",
            "password": "WrongPassword"
        })
        assert response.status_code == 401

class TestChannels:
    @pytest.fixture
    def auth_headers(self):
        response = client.post("/api/v1/auth/login", json={
            "email": "test@example.com",
            "password": "TestPassword123!"
        })
        token = response.json()["access_token"]
        return {"Authorization": f"Bearer {token}"}
    
    def test_create_channel(self, auth_headers):
        response = client.post("/api/v1/channels", 
            headers=auth_headers,
            json={
                "name": "Test Channel",
                "niche": "technology"
            })
        assert response.status_code == 201
        assert response.json()["name"] == "Test Channel"
    
    def test_list_channels(self, auth_headers):
        response = client.get("/api/v1/channels", headers=auth_headers)
        assert response.status_code == 200
        assert "channels" in response.json()
```

#### Frontend Testing (React/Vitest)
```typescript
// tests/Dashboard.test.tsx
import { render, screen, waitFor } from '@testing-library/react';
import { Dashboard } from '../src/pages/Dashboard';
import { mockApi } from './mocks/api';

describe('Dashboard', () => {
  beforeEach(() => {
    mockApi.reset();
  });

  test('renders dashboard metrics', async () => {
    mockApi.onGet('/api/v1/analytics/dashboard').reply(200, {
      summary: {
        total_channels: 5,
        total_videos: 100,
        total_revenue: 1234.56
      }
    });

    render(<Dashboard />);
    
    await waitFor(() => {
      expect(screen.getByText('5')).toBeInTheDocument();
      expect(screen.getByText('100')).toBeInTheDocument();
      expect(screen.getByText('$1,234.56')).toBeInTheDocument();
    });
  });

  test('handles loading state', () => {
    render(<Dashboard />);
    expect(screen.getByTestId('loading-spinner')).toBeInTheDocument();
  });

  test('handles error state', async () => {
    mockApi.onGet('/api/v1/analytics/dashboard').reply(500);

    render(<Dashboard />);
    
    await waitFor(() => {
      expect(screen.getByText('Failed to load dashboard')).toBeInTheDocument();
    });
  });
});
```

### Integration Testing

#### API Integration Tests
```python
# tests/integration/test_video_pipeline.py
import pytest
import asyncio
from app.services.video_service import VideoService

class TestVideoGeneration:
    @pytest.mark.asyncio
    async def test_full_video_generation_pipeline(self, db_session):
        # Setup
        video_service = VideoService(db_session)
        
        # Create video request
        video = await video_service.create_video({
            "channel_id": "test-channel",
            "topic": "Test Topic",
            "style": "educational"
        })
        
        # Process video
        await video_service.process_video(video.id)
        
        # Wait for completion
        max_attempts = 60
        for _ in range(max_attempts):
            video = await video_service.get_video(video.id)
            if video.status == "completed":
                break
            await asyncio.sleep(1)
        
        # Assertions
        assert video.status == "completed"
        assert video.youtube_video_id is not None
        assert video.cost < 3.00  # Cost constraint
        assert video.processing_time < 600  # 10 minute constraint
```

### End-to-End Testing

#### Playwright E2E Tests
```typescript
// e2e/channel-creation.spec.ts
import { test, expect } from '@playwright/test';

test.describe('Channel Creation Flow', () => {
  test('complete channel setup wizard', async ({ page }) => {
    // Login
    await page.goto('/login');
    await page.fill('[name="email"]', 'test@example.com');
    await page.fill('[name="password"]', 'TestPassword123!');
    await page.click('[type="submit"]');
    
    // Navigate to channels
    await page.waitForURL('/dashboard');
    await page.click('[data-testid="create-channel-button"]');
    
    // Step 1: Basic Information
    await page.fill('[name="channelName"]', 'E2E Test Channel');
    await page.selectOption('[name="niche"]', 'technology');
    