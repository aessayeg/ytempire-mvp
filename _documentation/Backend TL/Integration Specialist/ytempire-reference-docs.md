# 10. REFERENCE

## 10.1 API Reference

### Authentication

All API requests require authentication using JWT tokens.

#### Obtain Access Token

```http
POST /api/v1/auth/login
Content-Type: application/json

{
  "email": "user@example.com",
  "password": "secure_password"
}

Response:
{
  "access_token": "eyJhbGciOiJIUzI1NiIs...",
  "refresh_token": "eyJhbGciOiJIUzI1NiIs...",
  "expires_in": 900,
  "token_type": "Bearer"
}
```

#### Refresh Token

```http
POST /api/v1/auth/refresh
Content-Type: application/json

{
  "refresh_token": "eyJhbGciOiJIUzI1NiIs..."
}

Response:
{
  "access_token": "eyJhbGciOiJIUzI1NiIs...",
  "expires_in": 900
}
```

### Channels API

#### List Channels

```http
GET /api/v1/channels
Authorization: Bearer {access_token}

Response:
{
  "channels": [
    {
      "id": "uuid",
      "name": "TechReviewPro",
      "youtube_channel_id": "UC...",
      "niche": "Technology",
      "status": "active",
      "subscriber_count": 12500,
      "monthly_revenue": 2345.67,
      "health_score": 0.92
    }
  ],
  "total": 5,
  "page": 1,
  "per_page": 20
}
```

#### Create Channel

```http
POST /api/v1/channels
Authorization: Bearer {access_token}
Content-Type: application/json

{
  "name": "NewChannel",
  "niche": "Technology",
  "automation_enabled": true,
  "videos_per_day": 3,
  "publish_schedule": {
    "days": ["mon", "wed", "fri"],
    "times": ["09:00", "15:00"]
  }
}

Response:
{
  "id": "uuid",
  "name": "NewChannel",
  "status": "pending_auth",
  "oauth_url": "https://accounts.google.com/oauth/authorize?..."
}
```

#### Update Channel

```http
PUT /api/v1/channels/{channel_id}
Authorization: Bearer {access_token}
Content-Type: application/json

{
  "automation_enabled": false,
  "videos_per_day": 5,
  "quality_threshold": 0.85
}

Response:
{
  "id": "uuid",
  "updated_at": "2025-01-15T10:00:00Z",
  "changes_applied": ["automation_enabled", "videos_per_day", "quality_threshold"]
}
```

### Videos API

#### Generate Video

```http
POST /api/v1/videos/generate
Authorization: Bearer {access_token}
Content-Type: application/json

{
  "channel_id": "uuid",
  "topic": "iPhone 15 Review",
  "style": "educational",
  "length": 480,
  "priority": 5,
  "optimization_level": "standard"
}

Response:
{
  "video_id": "uuid",
  "status": "queued",
  "estimated_completion": "2025-01-15T10:30:00Z",
  "queue_position": 3
}
```

#### Get Video Status

```http
GET /api/v1/videos/{video_id}
Authorization: Bearer {access_token}

Response:
{
  "id": "uuid",
  "title": "iPhone 15 Review",
  "status": "processing",
  "progress": 65,
  "channel_id": "uuid",
  "youtube_video_id": null,
  "cost": 0.45,
  "created_at": "2025-01-15T10:00:00Z",
  "processing_started_at": "2025-01-15T10:05:00Z"
}
```

#### List Videos

```http
GET /api/v1/videos?channel_id={channel_id}&status=published&page=1&per_page=20
Authorization: Bearer {access_token}

Response:
{
  "videos": [
    {
      "id": "uuid",
      "title": "Video Title",
      "youtube_video_id": "dQw4w9WgXcQ",
      "status": "published",
      "views": 1234,
      "revenue": 12.34,
      "published_at": "2025-01-15T10:00:00Z"
    }
  ],
  "total": 145,
  "page": 1,
  "per_page": 20
}
```

### Analytics API

#### Get Channel Analytics

```http
GET /api/v1/analytics/channels/{channel_id}?period=30d
Authorization: Bearer {access_token}

Response:
{
  "channel_id": "uuid",
  "period": "30d",
  "metrics": {
    "views": 234567,
    "watch_time_hours": 3456,
    "subscribers_gained": 1234,
    "revenue": 3456.78,
    "videos_published": 90,
    "avg_views_per_video": 2606
  },
  "daily_breakdown": [
    {
      "date": "2025-01-15",
      "views": 8901,
      "revenue": 123.45
    }
  ]
}
```

#### Get Cost Analytics

```http
GET /api/v1/analytics/costs?period=7d
Authorization: Bearer {access_token}

Response:
{
  "period": "7d",
  "total_cost": 147.89,
  "total_videos": 350,
  "avg_cost_per_video": 0.42,
  "breakdown": {
    "openai": 89.45,
    "google_tts": 23.67,
    "processing": 17.89,
    "storage": 16.88
  },
  "daily_costs": [
    {
      "date": "2025-01-15",
      "cost": 21.13,
      "videos": 50
    }
  ]
}
```

### Webhooks API

#### Register Webhook

```http
POST /api/v1/webhooks
Authorization: Bearer {access_token}
Content-Type: application/json

{
  "url": "https://your-server.com/webhook",
  "events": ["video.completed", "video.failed", "channel.suspended"],
  "secret": "your_webhook_secret"
}

Response:
{
  "id": "webhook_uuid",
  "url": "https://your-server.com/webhook",
  "events": ["video.completed", "video.failed", "channel.suspended"],
  "status": "active",
  "created_at": "2025-01-15T10:00:00Z"
}
```

#### Webhook Event Payload

```json
{
  "id": "event_uuid",
  "type": "video.completed",
  "timestamp": "2025-01-15T10:30:00Z",
  "data": {
    "video_id": "uuid",
    "channel_id": "uuid",
    "youtube_video_id": "dQw4w9WgXcQ",
    "title": "Video Title",
    "cost": 0.42,
    "processing_time": 287
  },
  "signature": "sha256=..."
}
```

### Rate Limits

```yaml
Rate Limits by Plan:
  Starter:
    - 100 requests per minute
    - 15 video generations per day
    - 5 channels maximum
    
  Growth:
    - 300 requests per minute
    - 30 video generations per day
    - 10 channels maximum
    
  Scale:
    - 1000 requests per minute
    - 75 video generations per day
    - 25 channels maximum

Headers:
  X-RateLimit-Limit: 100
  X-RateLimit-Remaining: 95
  X-RateLimit-Reset: 1642255200
```

### Error Responses

```json
{
  "error": {
    "code": "RESOURCE_NOT_FOUND",
    "message": "The requested video was not found",
    "details": {
      "video_id": "invalid_uuid"
    },
    "request_id": "req_abc123"
  }
}
```

**Common Error Codes:**

| Code | HTTP Status | Description |
|------|-------------|-------------|
| AUTHENTICATION_REQUIRED | 401 | Missing or invalid token |
| INSUFFICIENT_PERMISSIONS | 403 | User lacks required permissions |
| RESOURCE_NOT_FOUND | 404 | Requested resource doesn't exist |
| VALIDATION_ERROR | 400 | Invalid request parameters |
| RATE_LIMIT_EXCEEDED | 429 | Too many requests |
| INTERNAL_ERROR | 500 | Server error |
| SERVICE_UNAVAILABLE | 503 | Temporary service issue |

## 10.2 Database Schema

### Core Tables

#### users.accounts

```sql
CREATE TABLE users.accounts (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    email VARCHAR(255) UNIQUE NOT NULL,
    email_verified BOOLEAN DEFAULT FALSE,
    password_hash VARCHAR(255) NOT NULL,
    full_name VARCHAR(255),
    company_name VARCHAR(255),
    phone VARCHAR(50),
    
    -- Subscription
    stripe_customer_id VARCHAR(255) UNIQUE,
    subscription_tier VARCHAR(50) DEFAULT 'free',
    subscription_status VARCHAR(50) DEFAULT 'inactive',
    subscription_id VARCHAR(255),
    trial_ends_at TIMESTAMP,
    
    -- Limits
    channel_limit INTEGER DEFAULT 5,
    daily_video_limit INTEGER DEFAULT 15,
    
    -- Status
    is_active BOOLEAN DEFAULT TRUE,
    is_beta_user BOOLEAN DEFAULT FALSE,
    onboarding_completed BOOLEAN DEFAULT FALSE,
    
    -- Timestamps
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    last_login_at TIMESTAMP WITH TIME ZONE
);
```

#### channels.youtube_channels

```sql
CREATE TABLE channels.youtube_channels (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    user_id UUID NOT NULL REFERENCES users.accounts(id) ON DELETE CASCADE,
    
    -- YouTube Info
    youtube_channel_id VARCHAR(255) UNIQUE NOT NULL,
    youtube_channel_handle VARCHAR(255),
    channel_title VARCHAR(255) NOT NULL,
    channel_description TEXT,
    
    -- Configuration
    niche VARCHAR(100) NOT NULL,
    sub_niche VARCHAR(100),
    target_audience JSONB,
    content_strategy JSONB,
    
    -- Automation
    automation_enabled BOOLEAN DEFAULT TRUE,
    auto_publish BOOLEAN DEFAULT FALSE,
    publish_schedule JSONB,
    videos_per_day INTEGER DEFAULT 3,
    
    -- Metrics
    subscriber_count INTEGER DEFAULT 0,
    total_views BIGINT DEFAULT 0,
    total_videos INTEGER DEFAULT 0,
    monetization_enabled BOOLEAN DEFAULT FALSE,
    estimated_monthly_revenue DECIMAL(10,2) DEFAULT 0.00,
    
    -- Status
    status VARCHAR(50) DEFAULT 'active',
    health_score DECIMAL(3,2) DEFAULT 1.00,
    last_video_at TIMESTAMP WITH TIME ZONE,
    
    -- OAuth
    oauth_credentials JSONB,
    oauth_refresh_token TEXT,
    oauth_expires_at TIMESTAMP WITH TIME ZONE,
    
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);
```

#### videos.video_records

```sql
CREATE TABLE videos.video_records (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    channel_id UUID NOT NULL REFERENCES channels.youtube_channels(id) ON DELETE CASCADE,
    user_id UUID NOT NULL REFERENCES users.accounts(id) ON DELETE CASCADE,
    
    -- Metadata
    title VARCHAR(500) NOT NULL,
    description TEXT,
    tags TEXT[],
    category_id VARCHAR(50),
    
    -- Content
    script TEXT,
    script_model VARCHAR(50),
    voice_provider VARCHAR(50),
    voice_id VARCHAR(100),
    
    -- Files
    video_file_path VARCHAR(500),
    thumbnail_file_path VARCHAR(500),
    audio_file_path VARCHAR(500),
    
    -- YouTube
    youtube_video_id VARCHAR(255) UNIQUE,
    youtube_url VARCHAR(500),
    youtube_upload_status VARCHAR(50),
    
    -- Processing
    status VARCHAR(50) DEFAULT 'queued',
    processing_started_at TIMESTAMP WITH TIME ZONE,
    processing_completed_at TIMESTAMP WITH TIME ZONE,
    processing_duration_seconds INTEGER,
    
    -- Quality
    quality_score DECIMAL(3,2),
    compliance_score DECIMAL(3,2),
    
    -- Scheduling
    scheduled_publish_at TIMESTAMP WITH TIME ZONE,
    published_at TIMESTAMP WITH TIME ZONE,
    
    -- Errors
    error_message TEXT,
    error_count INTEGER DEFAULT 0,
    last_error_at TIMESTAMP WITH TIME ZONE,
    
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);
```

#### costs.video_costs

```sql
CREATE TABLE costs.video_costs (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    video_id UUID NOT NULL REFERENCES videos.video_records(id) ON DELETE CASCADE,
    
    -- Service Costs
    script_generation_cost DECIMAL(10,4) DEFAULT 0.0000,
    voice_synthesis_cost DECIMAL(10,4) DEFAULT 0.0000,
    video_processing_cost DECIMAL(10,4) DEFAULT 0.0000,
    thumbnail_generation_cost DECIMAL(10,4) DEFAULT 0.0000,
    storage_cost DECIMAL(10,4) DEFAULT 0.0000,
    
    -- API Costs
    openai_cost DECIMAL(10,4) DEFAULT 0.0000,
    elevenlabs_cost DECIMAL(10,4) DEFAULT 0.0000,
    google_tts_cost DECIMAL(10,4) DEFAULT 0.0000,
    youtube_api_cost DECIMAL(10,4) DEFAULT 0.0000,
    
    -- Total
    total_cost DECIMAL(10,4) DEFAULT 0.0000,
    optimization_level VARCHAR(20) DEFAULT 'standard',
    
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);
```

### Indexes

```sql
-- Performance Indexes
CREATE INDEX idx_videos_channel ON videos.video_records(channel_id);
CREATE INDEX idx_videos_user ON videos.video_records(user_id);
CREATE INDEX idx_videos_status ON videos.video_records(status);
CREATE INDEX idx_videos_created ON videos.video_records(created_at DESC);

CREATE INDEX idx_channels_user ON channels.youtube_channels(user_id);
CREATE INDEX idx_channels_status ON channels.youtube_channels(status);

CREATE INDEX idx_costs_video ON costs.video_costs(video_id);
CREATE INDEX idx_costs_created ON costs.video_costs(created_at DESC);

-- Full-text Search
CREATE INDEX idx_videos_title_search ON videos.video_records USING gin(to_tsvector('english', title));
CREATE INDEX idx_videos_description_search ON videos.video_records USING gin(to_tsvector('english', description));
```

## 10.3 Configuration Files

### Docker Compose

```yaml
# docker-compose.yml
version: '3.8'

services:
  postgres:
    image: postgres:15
    environment:
      POSTGRES_DB: ytempire
      POSTGRES_USER: ${DB_USER}
      POSTGRES_PASSWORD: ${DB_PASSWORD}
    volumes:
      - postgres_data:/var/lib/postgresql/data
    ports:
      - "5432:5432"

  redis:
    image: redis:7-alpine
    command: redis-server --maxmemory 8gb --maxmemory-policy allkeys-lru
    ports:
      - "6379:6379"

  backend:
    build: ./backend
    environment:
      DATABASE_URL: postgresql://${DB_USER}:${DB_PASSWORD}@postgres:5432/ytempire
      REDIS_URL: redis://redis:6379
    depends_on:
      - postgres
      - redis
    ports:
      - "8000:8000"

  frontend:
    build: ./frontend
    environment:
      REACT_APP_API_URL: http://backend:8000
    ports:
      - "3000:3000"

  n8n:
    image: n8nio/n8n
    environment:
      - N8N_BASIC_AUTH_ACTIVE=true
      - N8N_BASIC_AUTH_USER=${N8N_USER}
      - N8N_BASIC_AUTH_PASSWORD=${N8N_PASSWORD}
    ports:
      - "5678:5678"
    volumes:
      - n8n_data:/home/node/.n8n

volumes:
  postgres_data:
  n8n_data:
```

### Environment Variables

```bash
# .env.example
# Database
DB_USER=ytempire
DB_PASSWORD=secure_password_here
DB_HOST=localhost
DB_PORT=5432
DB_NAME=ytempire

# Redis
REDIS_HOST=localhost
REDIS_PORT=6379
REDIS_PASSWORD=

# Security
JWT_SECRET=your_jwt_secret_key_here
ENCRYPTION_KEY=your_encryption_key_here

# APIs
OPENAI_API_KEY=sk-...
ELEVENLABS_API_KEY=...
GOOGLE_APPLICATION_CREDENTIALS=/path/to/credentials.json
STRIPE_SECRET_KEY=sk_live_...
STRIPE_WEBHOOK_SECRET=whsec_...

# YouTube (per account)
YOUTUBE_CLIENT_ID=...
YOUTUBE_CLIENT_SECRET=...

# N8N
N8N_USER=admin
N8N_PASSWORD=secure_password

# Application
NODE_ENV=production
LOG_LEVEL=info
PORT=8000
```

### Nginx Configuration

```nginx
# nginx.conf
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

    # Frontend
    location / {
        proxy_pass http://frontend:3000;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
    }

    # API
    location /api {
        proxy_pass http://backend:8000;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_read_timeout 300s;
    }

    # WebSocket
    location /ws {
        proxy_pass http://backend:8000;
        proxy_http_version 1.1;
        proxy_set_header Upgrade $http_upgrade;
        proxy_set_header Connection "upgrade";
    }

    # N8N
    location /workflows {
        proxy_pass http://n8n:5678;
        proxy_set_header Host $host;
    }
}
```

## 10.4 Glossary

### Business Terms

| Term | Definition |
|------|------------|
| **Channel** | A YouTube channel connected to YTEMPIRE platform |
| **Niche** | Specific content category or topic area (e.g., Technology, Finance) |
| **Health Score** | Metric (0-100) indicating channel performance and compliance |
| **Automation Level** | Percentage of tasks handled without human intervention |
| **Revenue Per Mille (RPM)** | Revenue per 1,000 video views |
| **Click-Through Rate (CTR)** | Percentage of impressions that result in clicks |
| **Watch Time** | Total minutes viewers spend watching videos |
| **Monetization** | Process of earning revenue from videos |

### Technical Terms

| Term | Definition |
|------|------------|
| **API** | Application Programming Interface for service communication |
| **JWT** | JSON Web Token used for authentication |
| **Webhook** | HTTP callback for real-time event notifications |
| **Queue** | System for managing video generation tasks |
| **Pipeline** | Series of processing stages for video creation |
| **Circuit Breaker** | Pattern to prevent cascading failures |
| **Rate Limiting** | Restricting number of API calls per time period |
| **Fallback** | Alternative service when primary fails |

### Platform Components

| Component | Description |
|-----------|-------------|
| **N8N** | Workflow automation platform |
| **FastAPI** | Python web framework for building APIs |
| **PostgreSQL** | Primary database system |
| **Redis** | In-memory cache and queue system |
| **Docker** | Container platform for deployment |
| **Prometheus** | Metrics collection system |
| **Grafana** | Metrics visualization dashboard |

### Cost Terms

| Term | Definition |
|------|------------|
| **Cost Per Video** | Total expenses to generate one video |
| **API Cost** | Charges from external services (OpenAI, etc.) |
| **Infrastructure Cost** | Server, storage, and bandwidth expenses |
| **Optimization Level** | Economy, Standard, or Premium processing |
| **Kill Switch** | Emergency stop when costs exceed limits |
| **Cache Hit Rate** | Percentage of requests served from cache |

## 10.5 Quick Reference Cards

### Daily Operations Checklist

```yaml
Morning (9:00 AM):
  System Health:
    □ Check overnight video generation
    □ Review error logs
    □ Verify all services running
    □ Check quota usage
    
  Content Review:
    □ Approve queued videos
    □ Review failed videos
    □ Check publishing schedule
    
  Cost Check:
    □ Yesterday's total cost
    □ Cost per video average
    □ Identify anomalies

Afternoon (2:00 PM):
  Performance:
    □ API response times
    □ Queue depth
    □ Processing times
    
  Monitoring:
    □ Channel health scores
    □ Upload success rate
    □ Revenue tracking

Evening (6:00 PM):
  Planning:
    □ Tomorrow's video schedule
    □ Resource allocation
    □ Cost projections
    
  Maintenance:
    □ Clear temp files
    □ Update documentation
    □ Backup verification
```

### Emergency Contacts

```yaml
Escalation Path:
  Level 1 (Immediate):
    - On-call Engineer: Check PagerDuty
    - Backup: Check team calendar
    
  Level 2 (15 minutes):
    - Backend Team Lead
    - Platform Ops Lead
    
  Level 3 (30 minutes):
    - CTO
    - VP of AI
    
  External Services:
    - YouTube API Support: youtube-api-support@google.com
    - OpenAI: support@openai.com
    - Stripe: support@stripe.com
    - AWS: Premium support console

System Access:
  - Production Server: ssh ytempire@prod.server
  - Database: psql -h localhost -U ytempire -d ytempire
  - Redis: redis-cli -h localhost
  - N8N: http://localhost:5678
  - Monitoring: http://localhost:3000/grafana
```

### Common Commands

```bash
# Docker Operations
docker-compose up -d              # Start all services
docker-compose down               # Stop all services
docker-compose logs -f backend    # View backend logs
docker-compose restart n8n        # Restart N8N

# Database Operations
pg_dump ytempire > backup.sql    # Backup database
psql ytempire < backup.sql       # Restore database
psql -c "VACUUM ANALYZE;"        # Optimize database

# Redis Operations
redis-cli FLUSHALL               # Clear all cache (CAUTION!)
redis-cli INFO memory            # Check memory usage
redis-cli MONITOR                # Watch commands in real-time

# System Monitoring
htop                             # CPU and memory usage
nvidia-smi                       # GPU status
df -h                           # Disk usage
netstat -tulpn                  # Network connections

# Log Analysis
tail -f /var/log/ytempire/app.log              # Application logs
grep ERROR /var/log/ytempire/app.log           # Find errors
journalctl -u ytempire-api -f                  # Service logs

# Cost Analysis
python scripts/cost_report.py --date today     # Today's costs
python scripts/optimize_costs.py --aggressive  # Apply optimizations
```

### Performance Targets

```yaml
System Performance:
  API Response: <500ms p95
  Video Generation: <10 minutes
  Dashboard Load: <2 seconds
  Queue Processing: <30 seconds
  Database Query: <150ms p95

Business Metrics:
  Cost per Video: <$1.00 (target)
  Upload Success: >99%
  Automation Rate: >95%
  Channel Health: >85
  User Satisfaction: >4.5/5

Scale Targets:
  MVP: 50 videos/day
  6 Months: 150 videos/day
  12 Months: 300+ videos/day
  Long-term: 3000+ videos/day
```