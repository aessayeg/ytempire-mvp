# YTEMPIRE Reference Documentation

## 8.1 API Documentation

### API Overview

**Base URL**: `https://api.ytempire.com/api/v1`  
**Protocol**: HTTPS only  
**Authentication**: JWT Bearer Token  
**Content-Type**: `application/json`  
**Rate Limiting**: 1000 requests/hour (default)

### Authentication

#### Register New User
```http
POST /auth/register

Request:
{
  "email": "user@example.com",
  "username": "johndoe",
  "password": "SecurePass123!",
  "terms_accepted": true
}

Response (201):
{
  "user_id": "550e8400-e29b-41d4-a716-446655440000",
  "email": "user@example.com",
  "username": "johndoe",
  "access_token": "eyJhbGciOiJIUzI1NiIs...",
  "refresh_token": "eyJhbGciOiJIUzI1NiIs...",
  "expires_in": 1800
}

Errors:
- 400: Invalid input data
- 409: Email already exists
```

#### Login
```http
POST /auth/login

Request:
{
  "email": "user@example.com",
  "password": "SecurePass123!"
}

Response (200):
{
  "access_token": "eyJhbGciOiJIUzI1NiIs...",
  "refresh_token": "eyJhbGciOiJIUzI1NiIs...",
  "token_type": "Bearer",
  "expires_in": 1800
}

Errors:
- 401: Invalid credentials
- 429: Too many login attempts
```

### Channel Management

#### List Channels
```http
GET /channels
Authorization: Bearer {token}

Query Parameters:
- page (int): Page number (default: 1)
- limit (int): Items per page (default: 10, max: 50)
- status (string): Filter by status (active|paused|deleted)

Response (200):
{
  "channels": [
    {
      "id": "550e8400-e29b-41d4-a716-446655440001",
      "name": "Tech Reviews",
      "youtube_channel_id": "UC_xyz123",
      "niche": "technology",
      "status": "active",
      "subscriber_count": 1543,
      "video_count": 47,
      "monetization_enabled": true,
      "created_at": "2025-01-01T00:00:00Z"
    }
  ],
  "pagination": {
    "page": 1,
    "limit": 10,
    "total": 3,
    "pages": 1
  }
}
```

#### Create Channel
```http
POST /channels
Authorization: Bearer {token}

Request:
{
  "name": "My Tech Channel",
  "niche": "technology",
  "youtube_channel_id": "UC_xyz123",
  "target_audience": {
    "age_range": "18-34",
    "interests": ["tech", "gadgets"],
    "geography": "US"
  }
}

Response (201):
{
  "id": "550e8400-e29b-41d4-a716-446655440002",
  "name": "My Tech Channel",
  "status": "active",
  "created_at": "2025-01-09T10:00:00Z"
}

Errors:
- 400: Invalid channel data
- 403: Channel limit exceeded
- 409: YouTube channel already linked
```

### Video Generation

#### Generate Video
```http
POST /videos/generate
Authorization: Bearer {token}

Request:
{
  "channel_id": "550e8400-e29b-41d4-a716-446655440001",
  "title": "Top 10 Gadgets 2025",
  "topic": "Review of the latest technology gadgets",
  "video_config": {
    "duration": 600,
    "voice": "professional-male",
    "language": "en-US",
    "style": "educational"
  },
  "monetization": {
    "enable_ads": true,
    "affiliate_links": true
  }
}

Response (202):
{
  "job_id": "job_550e8400-e29b-41d4",
  "video_id": "vid_550e8400-e29b-41d4",
  "status": "queued",
  "estimated_completion": "2025-01-09T10:10:00Z",
  "estimated_cost": 0.85,
  "queue_position": 3
}
```

#### Check Generation Status
```http
GET /videos/{video_id}/status
Authorization: Bearer {token}

Response (200):
{
  "video_id": "vid_550e8400-e29b-41d4",
  "status": "processing",
  "progress": 45,
  "current_stage": "voice_synthesis",
  "stages_completed": [
    "script_generation",
    "content_optimization"
  ],
  "stages_remaining": [
    "voice_synthesis",
    "video_assembly",
    "thumbnail_creation",
    "youtube_upload"
  ]
}
```

### Analytics

#### Get Analytics Overview
```http
GET /analytics/overview
Authorization: Bearer {token}

Query Parameters:
- start_date (ISO 8601): Start of period
- end_date (ISO 8601): End of period
- channel_id (string): Optional channel filter

Response (200):
{
  "period": {
    "start": "2025-01-01T00:00:00Z",
    "end": "2025-01-09T23:59:59Z"
  },
  "summary": {
    "total_views": 156789,
    "total_revenue": 1234.56,
    "subscriber_growth": 543,
    "videos_published": 47,
    "engagement_rate": 4.7
  },
  "channels": [
    {
      "channel_id": "550e8400-e29b-41d4-a716-446655440001",
      "views": 45678,
      "revenue": 345.67,
      "subscribers_gained": 123
    }
  ]
}
```

### Error Responses

```json
{
  "error": {
    "code": "VALIDATION_ERROR",
    "message": "Invalid request data",
    "details": [
      {
        "field": "email",
        "message": "Invalid email format"
      }
    ]
  },
  "request_id": "req_550e8400",
  "timestamp": "2025-01-09T10:00:00Z"
}
```

### Rate Limiting

Rate limit information is included in response headers:
```
X-RateLimit-Limit: 1000
X-RateLimit-Remaining: 998
X-RateLimit-Reset: 1704796800
```

## 8.2 Database Schema

### Core Tables

#### users
```sql
CREATE TABLE users (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    email VARCHAR(255) UNIQUE NOT NULL,
    username VARCHAR(50) UNIQUE NOT NULL,
    password_hash VARCHAR(255) NOT NULL,
    subscription_tier VARCHAR(20) DEFAULT 'free',
    subscription_expires_at TIMESTAMPTZ,
    created_at TIMESTAMPTZ DEFAULT NOW(),
    updated_at TIMESTAMPTZ DEFAULT NOW(),
    last_login TIMESTAMPTZ,
    status VARCHAR(20) DEFAULT 'active',
    metadata JSONB DEFAULT '{}',
    
    CONSTRAINT email_format CHECK (email ~* '^[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}$'),
    CONSTRAINT username_length CHECK (LENGTH(username) >= 3 AND LENGTH(username) <= 50),
    CONSTRAINT subscription_tier_valid CHECK (subscription_tier IN ('free', 'starter', 'professional', 'enterprise'))
);

CREATE INDEX idx_users_email ON users(email);
CREATE INDEX idx_users_status ON users(status);
CREATE INDEX idx_users_subscription ON users(subscription_tier);
```

#### channels
```sql
CREATE TABLE channels (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    user_id UUID NOT NULL REFERENCES users(id) ON DELETE CASCADE,
    youtube_channel_id VARCHAR(100) UNIQUE NOT NULL,
    channel_name VARCHAR(255) NOT NULL,
    niche VARCHAR(100) NOT NULL,
    status VARCHAR(20) DEFAULT 'active',
    monetization_status VARCHAR(20) DEFAULT 'pending',
    subscriber_count INTEGER DEFAULT 0,
    total_views BIGINT DEFAULT 0,
    created_at TIMESTAMPTZ DEFAULT NOW(),
    updated_at TIMESTAMPTZ DEFAULT NOW(),
    settings JSONB DEFAULT '{}',
    metadata JSONB DEFAULT '{}',
    
    CONSTRAINT status_valid CHECK (status IN ('active', 'paused', 'deleted')),
    CONSTRAINT monetization_valid CHECK (monetization_status IN ('pending', 'eligible', 'enabled', 'suspended'))
);

CREATE INDEX idx_channels_user ON channels(user_id);
CREATE INDEX idx_channels_status ON channels(status);
CREATE INDEX idx_channels_youtube ON channels(youtube_channel_id);
```

#### videos
```sql
CREATE TABLE videos (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    channel_id UUID NOT NULL REFERENCES channels(id) ON DELETE CASCADE,
    title VARCHAR(500) NOT NULL,
    description TEXT,
    script TEXT,
    youtube_video_id VARCHAR(100) UNIQUE,
    thumbnail_url VARCHAR(500),
    status VARCHAR(20) DEFAULT 'draft',
    generation_cost DECIMAL(10,2),
    processing_time_seconds INTEGER,
    created_at TIMESTAMPTZ DEFAULT NOW(),
    published_at TIMESTAMPTZ,
    scheduled_at TIMESTAMPTZ,
    view_count BIGINT DEFAULT 0,
    like_count INTEGER DEFAULT 0,
    comment_count INTEGER DEFAULT 0,
    estimated_revenue DECIMAL(10,2) DEFAULT 0,
    metadata JSONB DEFAULT '{}',
    
    CONSTRAINT status_valid CHECK (status IN ('draft', 'queued', 'processing', 'completed', 'published', 'failed')),
    CONSTRAINT cost_positive CHECK (generation_cost >= 0)
);

CREATE INDEX idx_videos_channel ON videos(channel_id);
CREATE INDEX idx_videos_status ON videos(status);
CREATE INDEX idx_videos_published ON videos(published_at);
CREATE INDEX idx_videos_youtube ON videos(youtube_video_id);
```

#### generation_jobs
```sql
CREATE TABLE generation_jobs (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    video_id UUID REFERENCES videos(id) ON DELETE CASCADE,
    status VARCHAR(20) DEFAULT 'pending',
    progress INTEGER DEFAULT 0,
    current_stage VARCHAR(50),
    started_at TIMESTAMPTZ,
    completed_at TIMESTAMPTZ,
    error_message TEXT,
    error_details JSONB,
    cost_breakdown JSONB DEFAULT '{}',
    worker_id VARCHAR(100),
    retry_count INTEGER DEFAULT 0,
    max_retries INTEGER DEFAULT 3,
    
    CONSTRAINT status_valid CHECK (status IN ('pending', 'running', 'completed', 'failed', 'cancelled')),
    CONSTRAINT progress_valid CHECK (progress >= 0 AND progress <= 100)
);

CREATE INDEX idx_jobs_status ON generation_jobs(status);
CREATE INDEX idx_jobs_video ON generation_jobs(video_id);
CREATE INDEX idx_jobs_worker ON generation_jobs(worker_id);
```

### Database Functions

```sql
-- Function to update updated_at timestamp
CREATE OR REPLACE FUNCTION update_updated_at_column()
RETURNS TRIGGER AS $$
BEGIN
    NEW.updated_at = NOW();
    RETURN NEW;
END;
$$ language 'plpgsql';

-- Apply to all tables with updated_at
CREATE TRIGGER update_users_updated_at BEFORE UPDATE ON users
    FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();

CREATE TRIGGER update_channels_updated_at BEFORE UPDATE ON channels
    FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();

-- Function to enforce channel limits
CREATE OR REPLACE FUNCTION check_channel_limit()
RETURNS TRIGGER AS $$
DECLARE
    user_tier VARCHAR(20);
    current_count INTEGER;
    max_channels INTEGER;
BEGIN
    SELECT subscription_tier INTO user_tier
    FROM users WHERE id = NEW.user_id;
    
    SELECT COUNT(*) INTO current_count
    FROM channels 
    WHERE user_id = NEW.user_id AND status = 'active';
    
    max_channels := CASE user_tier
        WHEN 'free' THEN 1
        WHEN 'starter' THEN 5
        WHEN 'professional' THEN 20
        WHEN 'enterprise' THEN 1000
        ELSE 1
    END;
    
    IF current_count >= max_channels THEN
        RAISE EXCEPTION 'Channel limit exceeded for subscription tier %', user_tier;
    END IF;
    
    RETURN NEW;
END;
$$ language 'plpgsql';

CREATE TRIGGER enforce_channel_limit BEFORE INSERT ON channels
    FOR EACH ROW EXECUTE FUNCTION check_channel_limit();
```

## 8.3 Configuration Guide

### Environment Variables

```bash
# .env.example - Environment configuration template

# Application
APP_NAME=YTEMPIRE
APP_ENV=production
APP_DEBUG=false
APP_URL=https://ytempire.com
SECRET_KEY=your-secret-key-here

# Database
DATABASE_URL=postgresql://user:password@localhost:5432/ytempire
DB_POOL_SIZE=20
DB_MAX_OVERFLOW=40

# Redis
REDIS_URL=redis://localhost:6379/0
REDIS_MAX_CONNECTIONS=50

# Authentication
JWT_SECRET_KEY=your-jwt-secret
JWT_EXPIRATION_MINUTES=30
JWT_REFRESH_EXPIRATION_DAYS=7

# YouTube API
YOUTUBE_API_KEY=your-youtube-api-key
YOUTUBE_CLIENT_ID=your-client-id
YOUTUBE_CLIENT_SECRET=your-client-secret
YOUTUBE_QUOTA_PER_ACCOUNT=10000

# OpenAI
OPENAI_API_KEY=your-openai-key
OPENAI_MODEL=gpt-3.5-turbo
OPENAI_MAX_TOKENS=2000

# ElevenLabs
ELEVENLABS_API_KEY=your-elevenlabs-key
ELEVENLABS_VOICE_ID=voice-id

# Stripe
STRIPE_SECRET_KEY=your-stripe-secret
STRIPE_WEBHOOK_SECRET=your-webhook-secret

# Storage
STORAGE_TYPE=local
STORAGE_PATH=/data/videos
STORAGE_MAX_SIZE_GB=6000

# Monitoring
PROMETHEUS_PORT=9090
GRAFANA_PORT=3000
ALERT_EMAIL=alerts@ytempire.com
SLACK_WEBHOOK_URL=https://hooks.slack.com/...

# Limits
MAX_VIDEO_DURATION_SECONDS=1200
MAX_VIDEOS_PER_DAY_FREE=10
MAX_VIDEOS_PER_DAY_PRO=50
MAX_COST_PER_VIDEO=3.00
```

### Docker Configuration

```dockerfile
# Dockerfile - Application container

FROM python:3.11-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    gcc \
    postgresql-client \
    ffmpeg \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

# Create non-root user
RUN useradd -m -u 1000 ytempire && chown -R ytempire:ytempire /app
USER ytempire

# Health check
HEALTHCHECK --interval=30s --timeout=3s --start-period=5s --retries=3 \
    CMD python -c "import requests; requests.get('http://localhost:8000/health')"

# Run application
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
```

### Nginx Configuration

```nginx
# nginx.conf - Reverse proxy configuration

upstream backend {
    server app:8000;
}

server {
    listen 80;
    server_name ytempire.com;
    return 301 https://$server_name$request_uri;
}

server {
    listen 443 ssl http2;
    server_name ytempire.com;

    ssl_certificate /etc/nginx/ssl/cert.pem;
    ssl_certificate_key /etc/nginx/ssl/key.pem;
    ssl_protocols TLSv1.2 TLSv1.3;
    ssl_ciphers HIGH:!aNULL:!MD5;

    client_max_body_size 100M;
    
    location / {
        proxy_pass http://backend;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;
        
        # WebSocket support
        proxy_http_version 1.1;
        proxy_set_header Upgrade $http_upgrade;
        proxy_set_header Connection "upgrade";
    }
    
    location /static {
        alias /app/static;
        expires 30d;
        add_header Cache-Control "public, immutable";
    }
}
```

## 8.4 Troubleshooting Guide

### Common Issues and Solutions

#### Application Issues

**Issue: API returns 500 errors**
```bash
# Check application logs
docker-compose logs app --tail=100

# Common causes:
# 1. Database connection issues
docker-compose exec app python -c "from app.database import test_connection; test_connection()"

# 2. Redis connection issues
docker-compose exec redis redis-cli ping

# 3. Memory issues
docker stats

# Solution: Restart affected service
docker-compose restart app
```

**Issue: Video generation stuck**
```bash
# Check job status
docker-compose exec app python manage.py check_job <job_id>

# Check worker logs
docker-compose logs celery --tail=100

# Clear stuck jobs
docker-compose exec app python manage.py clear_stuck_jobs

# Restart workers
docker-compose restart celery
```

**Issue: High memory usage**
```bash
# Identify memory consumers
ps aux --sort=-%mem | head

# Check for memory leaks
docker-compose exec app python -m memory_profiler app/main.py

# Clear caches
docker-compose exec redis redis-cli FLUSHDB

# Restart services
docker-compose restart
```

#### Database Issues

**Issue: Slow queries**
```sql
-- Identify slow queries
SELECT query, calls, mean_time, max_time
FROM pg_stat_statements
WHERE mean_time > 100
ORDER BY mean_time DESC
LIMIT 10;

-- Check missing indexes
SELECT schemaname, tablename, attname, n_distinct, correlation
FROM pg_stats
WHERE schemaname = 'public'
AND n_distinct > 100
AND correlation < 0.1;

-- Add missing index
CREATE INDEX CONCURRENTLY idx_videos_created_at ON videos(created_at);
```

**Issue: Database connection pool exhausted**
```bash
# Check current connections
docker-compose exec postgres psql -U postgres -c "SELECT count(*) FROM pg_stat_activity;"

# Kill idle connections
docker-compose exec postgres psql -U postgres -c "
SELECT pg_terminate_backend(pid) 
FROM pg_stat_activity 
WHERE state = 'idle' 
AND state_change < now() - interval '10 minutes';"

# Increase pool size (temporarily)
export DB_POOL_SIZE=40
docker-compose restart app
```

#### Performance Issues

**Issue: High CPU usage**
```bash
# Identify CPU consumers
top -bn1 | head -20

# Check for runaway processes
ps aux | grep python

# Profile application
docker-compose exec app python -m cProfile -o profile.out app/main.py

# Analyze profile
docker-compose exec app python -m pstats profile.out
```

**Issue: Disk space issues**
```bash
# Check disk usage
df -h

# Find large files
find / -size +1G -type f 2>/dev/null

# Clean Docker resources
docker system prune -af --volumes

# Clean old logs
find /var/log -type f -name "*.log" -mtime +7 -delete

# Clean old videos
find /data/videos -type f -mtime +30 -delete
```

### Health Checks

```bash
#!/bin/bash
# health-check.sh - System health verification

echo "Running system health check..."

# Check services
services=("nginx" "app" "postgres" "redis" "celery")
for service in "${services[@]}"; do
    if docker-compose ps | grep -q "${service}.*Up"; then
        echo "✓ ${service} is healthy"
    else
        echo "✗ ${service} is down!"
        exit 1
    fi
done

# Check API
if curl -f http://localhost/health > /dev/null 2>&1; then
    echo "✓ API is responding"
else
    echo "✗ API is not responding!"
    exit 1
fi

# Check database
if docker-compose exec -T postgres psql -U postgres -c "SELECT 1" > /dev/null 2>&1; then
    echo "✓ Database is accessible"
else
    echo "✗ Database is not accessible!"
    exit 1
fi

echo "System health check completed!"
```

### Monitoring Commands

```bash
# Real-time logs
docker-compose logs -f app

# Resource usage
docker stats

# Database queries
docker-compose exec postgres psql -U postgres -c "SELECT * FROM pg_stat_activity;"

# Redis monitoring
docker-compose exec redis redis-cli monitor

# Queue status
docker-compose exec app celery -A app.celery inspect active

# API metrics
curl http://localhost:9090/metrics
```

---

*Document Status: Version 1.0 - January 2025*
*Owner: Platform Operations Team*
*Review Cycle: As needed*