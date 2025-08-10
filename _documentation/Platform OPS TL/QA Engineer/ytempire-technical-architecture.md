# YTEMPIRE Technical Architecture

## 4.1 System Architecture Overview

### High-Level Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                        Client Layer                          │
│  ┌──────────┐  ┌──────────┐  ┌──────────┐  ┌──────────┐  │
│  │    Web   │  │  Mobile  │  │    API   │  │   Admin  │  │
│  │    App   │  │    Web   │  │  Clients │  │   Panel  │  │
│  └──────────┘  └──────────┘  └──────────┘  └──────────┘  │
└─────────────────────────────────────────────────────────────┘
                                │
                                ▼
┌─────────────────────────────────────────────────────────────┐
│                     API Gateway Layer                        │
│  ┌──────────────────────────────────────────────────────┐  │
│  │          FastAPI Monolith (MVP Architecture)         │  │
│  │  ┌──────┐ ┌──────┐ ┌──────┐ ┌──────┐ ┌──────────┐ │  │
│  │  │ Auth │ │Video │ │Channel│ │Analytics│ │Webhooks│ │  │
│  │  └──────┘ └──────┘ └──────┘ └──────┘ └──────────┘ │  │
│  └──────────────────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────────────┘
                                │
                ┌───────────────┼───────────────┐
                ▼               ▼               ▼
┌──────────────────┐ ┌──────────────┐ ┌──────────────────┐
│  Business Logic  │ │   Queue      │ │   Cache Layer    │
│   Service Layer  │ │   (Celery)   │ │    (Redis)       │
└──────────────────┘ └──────────────┘ └──────────────────┘
                ▼               ▼               ▼
┌─────────────────────────────────────────────────────────────┐
│                     Data Layer                              │
│  ┌──────────────┐  ┌──────────────┐  ┌────────────────┐   │
│  │ PostgreSQL   │  │  File Storage│  │  Time Series   │   │
│  │  (Primary)   │  │    (NVMe)    │  │   (Metrics)    │   │
│  └──────────────┘  └──────────────┘  └────────────────┘   │
└─────────────────────────────────────────────────────────────┘
                                │
                                ▼
┌─────────────────────────────────────────────────────────────┐
│                    External Services                         │
│  ┌──────────┐ ┌──────────┐ ┌──────────┐ ┌──────────────┐  │
│  │ YouTube  │ │  OpenAI  │ │ElevenLabs│ │    Stripe    │  │
│  │   API    │ │   API    │ │   API    │ │   Payments   │  │
│  └──────────┘ └──────────┘ └──────────┘ └──────────────┘  │
└─────────────────────────────────────────────────────────────┘
```

### Component Architecture

#### Presentation Tier
- **Web Application**: React 18 SPA
- **Mobile Web**: Responsive design
- **Admin Panel**: Internal management
- **API Clients**: Third-party integrations

#### Application Tier
- **API Gateway**: FastAPI monolith
- **Business Logic**: Service layer pattern
- **Queue System**: Celery + Redis
- **Workflow Engine**: N8N automation

#### Data Tier
- **Primary Database**: PostgreSQL 15
- **Cache**: Redis 7
- **File Storage**: Local NVMe
- **Metrics**: Prometheus TSDB

#### Integration Tier
- **YouTube**: Content publishing
- **AI Services**: Content generation
- **Payment**: Stripe processing
- **Analytics**: External tracking

### Deployment Architecture

```
Local Server (MVP Deployment)
├── Hardware Layer
│   ├── CPU: AMD Ryzen 9 9950X3D (16 cores)
│   ├── RAM: 128GB DDR5
│   ├── GPU: NVIDIA RTX 5090 (32GB VRAM)
│   └── Storage: 10TB (2TB NVMe + 8TB SSD)
│
├── Operating System
│   └── Ubuntu 22.04 LTS
│
├── Container Runtime
│   └── Docker + Docker Compose
│
└── Service Containers
    ├── nginx (Reverse Proxy)
    ├── fastapi (API Server)
    ├── celery (Workers)
    ├── postgres (Database)
    ├── redis (Cache/Queue)
    ├── n8n (Workflow)
    └── prometheus/grafana (Monitoring)
```

## 4.2 Technology Stack

### Backend Technologies

#### Core Framework
**FastAPI** (Python 3.11+)
- Async/await support
- Automatic OpenAPI documentation
- Type hints and validation
- High performance
- WebSocket support

#### Database & Storage
**PostgreSQL 15**
- ACID compliance
- JSON/JSONB support
- Full-text search
- Partitioning capabilities
- Extensions (UUID, etc.)

**Redis 7**
- Session management
- Caching layer
- Message queue
- Rate limiting
- Real-time features

#### Queue & Processing
**Celery**
- Distributed task queue
- Scheduled tasks
- Priority queues
- Result backend
- Monitoring support

**N8N**
- Visual workflow automation
- Webhook handling
- Integration orchestration
- Error handling
- Monitoring

### Frontend Technologies

#### Core Framework
**React 18**
- Component architecture
- Hooks and context
- Suspense for data fetching
- Concurrent features
- Server components ready

#### State Management
**Redux Toolkit**
- Predictable state
- DevTools integration
- RTK Query for API
- Optimistic updates
- Cache management

#### UI Framework
**Material-UI v5**
- Component library
- Theming system
- Responsive design
- Accessibility built-in
- Customization options

**Tailwind CSS**
- Utility-first CSS
- Responsive utilities
- Dark mode support
- Performance optimized
- Component patterns

#### Build Tools
**Vite**
- Fast HMR
- Optimized builds
- ESM support
- Plugin ecosystem
- TypeScript support

### DevOps & Infrastructure

#### Containerization
**Docker**
- Container runtime
- Multi-stage builds
- Layer caching
- Security scanning
- Registry management

**Docker Compose**
- Service orchestration
- Environment management
- Network configuration
- Volume management
- Development workflow

#### CI/CD
**GitHub Actions**
- Automated testing
- Build pipeline
- Deployment automation
- Security scanning
- Release management

#### Monitoring
**Prometheus**
- Metrics collection
- Time-series database
- Alert rules
- Service discovery
- PromQL queries

**Grafana**
- Dashboard visualization
- Alert management
- Log exploration
- Performance monitoring
- Custom panels

### AI/ML Stack

#### Language Models
**OpenAI GPT-3.5/4**
- Script generation
- Content optimization
- Title creation
- Description writing
- Tag generation

#### Voice Synthesis
**ElevenLabs**
- Natural voices
- Multiple accents
- Emotion control
- Speed adjustment
- Batch processing

**Google Text-to-Speech**
- Fallback option
- Cost-effective
- Multiple languages
- SSML support
- Neural voices

#### Video Processing
**FFmpeg**
- Video encoding
- Audio processing
- Format conversion
- Thumbnail extraction
- Streaming support

**OpenCV**
- Image processing
- Video analysis
- Face detection
- Scene detection
- Quality assessment

## 4.3 Database Design

### Core Schema

```sql
-- Users and Authentication
CREATE TABLE users (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    email VARCHAR(255) UNIQUE NOT NULL,
    username VARCHAR(50) UNIQUE NOT NULL,
    password_hash VARCHAR(255) NOT NULL,
    subscription_tier VARCHAR(20) DEFAULT 'free',
    created_at TIMESTAMPTZ DEFAULT NOW(),
    updated_at TIMESTAMPTZ DEFAULT NOW(),
    last_login TIMESTAMPTZ,
    status VARCHAR(20) DEFAULT 'active',
    metadata JSONB DEFAULT '{}'
);

-- YouTube Channels
CREATE TABLE channels (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    user_id UUID NOT NULL REFERENCES users(id) ON DELETE CASCADE,
    youtube_channel_id VARCHAR(100) UNIQUE NOT NULL,
    channel_name VARCHAR(255) NOT NULL,
    niche VARCHAR(100) NOT NULL,
    status VARCHAR(20) DEFAULT 'active',
    monetization_status VARCHAR(20) DEFAULT 'pending',
    subscriber_count INTEGER DEFAULT 0,
    created_at TIMESTAMPTZ DEFAULT NOW(),
    updated_at TIMESTAMPTZ DEFAULT NOW(),
    settings JSONB DEFAULT '{}',
    INDEX idx_user_channels (user_id),
    INDEX idx_channel_status (status)
);

-- Videos
CREATE TABLE videos (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    channel_id UUID NOT NULL REFERENCES channels(id) ON DELETE CASCADE,
    title VARCHAR(500) NOT NULL,
    description TEXT,
    script TEXT,
    youtube_video_id VARCHAR(100) UNIQUE,
    status VARCHAR(20) DEFAULT 'draft',
    generation_cost DECIMAL(10,2),
    processing_time INTEGER,
    created_at TIMESTAMPTZ DEFAULT NOW(),
    published_at TIMESTAMPTZ,
    view_count BIGINT DEFAULT 0,
    like_count INTEGER DEFAULT 0,
    comment_count INTEGER DEFAULT 0,
    estimated_revenue DECIMAL(10,2) DEFAULT 0,
    metadata JSONB DEFAULT '{}',
    INDEX idx_channel_videos (channel_id),
    INDEX idx_video_status (status),
    INDEX idx_published_date (published_at)
);

-- Video Generation Jobs
CREATE TABLE generation_jobs (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    video_id UUID REFERENCES videos(id) ON DELETE CASCADE,
    status VARCHAR(20) DEFAULT 'pending',
    progress INTEGER DEFAULT 0,
    started_at TIMESTAMPTZ,
    completed_at TIMESTAMPTZ,
    error_message TEXT,
    cost_breakdown JSONB DEFAULT '{}',
    worker_id VARCHAR(100),
    retry_count INTEGER DEFAULT 0,
    INDEX idx_job_status (status),
    INDEX idx_job_video (video_id)
);

-- Analytics
CREATE TABLE analytics (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    video_id UUID NOT NULL REFERENCES videos(id) ON DELETE CASCADE,
    date DATE NOT NULL,
    views INTEGER DEFAULT 0,
    watch_time_minutes DECIMAL(10,2) DEFAULT 0,
    impressions INTEGER DEFAULT 0,
    click_through_rate DECIMAL(5,2) DEFAULT 0,
    average_view_duration INTEGER DEFAULT 0,
    subscribers_gained INTEGER DEFAULT 0,
    revenue DECIMAL(10,2) DEFAULT 0,
    UNIQUE(video_id, date),
    INDEX idx_analytics_date (date),
    INDEX idx_video_analytics (video_id)
);

-- API Keys and Integrations
CREATE TABLE api_keys (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    user_id UUID NOT NULL REFERENCES users(id) ON DELETE CASCADE,
    service_name VARCHAR(100) NOT NULL,
    api_key VARCHAR(500) NOT NULL,
    api_secret VARCHAR(500),
    status VARCHAR(20) DEFAULT 'active',
    created_at TIMESTAMPTZ DEFAULT NOW(),
    last_used TIMESTAMPTZ,
    usage_count INTEGER DEFAULT 0,
    metadata JSONB DEFAULT '{}',
    INDEX idx_user_keys (user_id),
    INDEX idx_service (service_name)
);

-- Workflow Templates
CREATE TABLE workflow_templates (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    name VARCHAR(255) NOT NULL,
    description TEXT,
    workflow_json JSONB NOT NULL,
    category VARCHAR(100),
    is_public BOOLEAN DEFAULT false,
    created_by UUID REFERENCES users(id),
    created_at TIMESTAMPTZ DEFAULT NOW(),
    updated_at TIMESTAMPTZ DEFAULT NOW(),
    usage_count INTEGER DEFAULT 0,
    INDEX idx_template_category (category),
    INDEX idx_template_public (is_public)
);
```

### Database Optimization

#### Indexing Strategy
- Primary keys on all tables
- Foreign key indexes for joins
- Composite indexes for common queries
- Partial indexes for filtered queries
- JSONB GIN indexes for metadata

#### Partitioning Strategy
- Analytics table by month
- Videos table by created_at
- Jobs table by status
- Automatic partition management

#### Performance Tuning
```sql
-- Connection pooling
max_connections = 200
shared_buffers = 32GB
effective_cache_size = 96GB
work_mem = 100MB
maintenance_work_mem = 2GB

-- Query optimization
random_page_cost = 1.1
effective_io_concurrency = 200
default_statistics_target = 100
```

## 4.4 API Architecture

### RESTful API Design

#### API Structure
```
/api/v1
├── /auth
│   ├── POST   /register
│   ├── POST   /login
│   ├── POST   /logout
│   ├── POST   /refresh
│   └── POST   /reset-password
│
├── /users
│   ├── GET    /profile
│   ├── PUT    /profile
│   ├── DELETE /account
│   └── GET    /subscription
│
├── /channels
│   ├── GET    /           (list channels)
│   ├── POST   /           (create channel)
│   ├── GET    /{id}       (get channel)
│   ├── PUT    /{id}       (update channel)
│   ├── DELETE /{id}       (delete channel)
│   └── POST   /{id}/sync  (sync with YouTube)
│
├── /videos
│   ├── GET    /           (list videos)
│   ├── POST   /generate   (generate video)
│   ├── GET    /{id}       (get video)
│   ├── PUT    /{id}       (update video)
│   ├── DELETE /{id}       (delete video)
│   ├── GET    /{id}/status (generation status)
│   └── POST   /{id}/publish (publish to YouTube)
│
├── /analytics
│   ├── GET    /overview   (dashboard metrics)
│   ├── GET    /channels   (channel analytics)
│   ├── GET    /videos     (video analytics)
│   └── GET    /revenue    (revenue reports)
│
└── /webhooks
    ├── POST   /stripe     (payment webhooks)
    ├── POST   /youtube    (YouTube webhooks)
    └── POST   /n8n        (workflow webhooks)
```

#### API Standards

**Request/Response Format**
```json
// Request
{
  "data": {
    "type": "video",
    "attributes": {
      "title": "Video Title",
      "channel_id": "uuid"
    }
  }
}

// Success Response
{
  "data": {
    "id": "uuid",
    "type": "video",
    "attributes": {}
  },
  "meta": {
    "timestamp": "2025-01-09T10:00:00Z"
  }
}

// Error Response
{
  "errors": [{
    "status": "400",
    "title": "Bad Request",
    "detail": "Invalid channel_id"
  }]
}
```

#### Authentication
- JWT tokens with refresh mechanism
- Token expiry: 30 minutes
- Refresh token: 7 days
- Rate limiting per user
- API key support for automation

### WebSocket Architecture

#### Real-time Events
```javascript
// WebSocket events
{
  "video.generation.started": {
    "video_id": "uuid",
    "timestamp": "ISO-8601"
  },
  "video.generation.progress": {
    "video_id": "uuid",
    "progress": 45,
    "stage": "voice_synthesis"
  },
  "video.generation.completed": {
    "video_id": "uuid",
    "youtube_url": "https://..."
  },
  "analytics.update": {
    "channel_id": "uuid",
    "metrics": {}
  }
}
```

## 4.5 Infrastructure Design

### Local Server Configuration

#### Hardware Allocation
```yaml
Resource Distribution:
  CPU (16 cores):
    - PostgreSQL: 4 cores
    - Backend API: 4 cores
    - Video Processing: 4 cores
    - Frontend/N8N: 2 cores
    - System/Monitoring: 2 cores
    
  Memory (128GB):
    - PostgreSQL: 32GB
    - Redis: 8GB
    - Backend Services: 24GB
    - Video Processing: 40GB
    - Frontend/N8N: 16GB
    - System/Buffer: 8GB
    
  Storage (10TB):
    - System/Apps: 500GB (NVMe)
    - Database: 1.5TB (NVMe)
    - Video Cache: 6TB (SSD)
    - Backups: 2TB (SSD)
    
  GPU (RTX 5090):
    - Video Rendering: 60%
    - AI Inference: 30%
    - Buffer: 10%
```

#### Network Architecture
```yaml
Network Configuration:
  External:
    - 1Gbps Fiber Connection
    - Static IP Address
    - DDoS Protection
    
  Internal:
    - Docker Bridge Network
    - Service Mesh
    - Internal DNS
    
  Security:
    - UFW Firewall
    - Fail2ban
    - VPN Access
```

#### Container Orchestration
```yaml
version: '3.8'

services:
  nginx:
    image: nginx:alpine
    ports:
      - "80:80"
      - "443:443"
    volumes:
      - ./nginx.conf:/etc/nginx/nginx.conf
      - ./ssl:/etc/nginx/ssl
    
  backend:
    build: ./backend
    environment:
      - DATABASE_URL=postgresql://...
      - REDIS_URL=redis://redis:6379
    depends_on:
      - postgres
      - redis
    
  postgres:
    image: postgres:15
    volumes:
      - postgres_data:/var/lib/postgresql/data
    environment:
      - POSTGRES_DB=ytempire
      - POSTGRES_USER=admin
      - POSTGRES_PASSWORD=${DB_PASSWORD}
    
  redis:
    image: redis:7-alpine
    command: redis-server --appendonly yes
    volumes:
      - redis_data:/data
    
  celery:
    build: ./backend
    command: celery worker -A app.celery -l info
    depends_on:
      - redis
      - postgres
```

## 4.6 Security Architecture

### Security Layers

#### Application Security
- **Authentication**: JWT with refresh tokens
- **Authorization**: Role-based access control
- **Input Validation**: Pydantic models
- **SQL Injection**: Parameterized queries
- **XSS Prevention**: Content security policy
- **CSRF Protection**: Token validation

#### Infrastructure Security
- **Network**: Firewall rules (UFW)
- **SSH**: Key-only authentication
- **SSL/TLS**: Let's Encrypt certificates
- **Secrets**: Environment variables
- **Monitoring**: Intrusion detection
- **Backups**: Encrypted storage

#### Data Security
- **Encryption at Rest**: Database encryption
- **Encryption in Transit**: TLS 1.3
- **Password Storage**: Bcrypt hashing
- **PII Protection**: Data minimization
- **Audit Logging**: All access logged
- **GDPR Compliance**: Data controls

### Security Protocols

#### Incident Response
```yaml
Detection:
  - Automated monitoring
  - Anomaly detection
  - User reports
  
Response:
  - Immediate isolation
  - Evidence collection
  - Root cause analysis
  
Recovery:
  - System restoration
  - Security patches
  - Post-mortem review
  
Prevention:
  - Security updates
  - Access review
  - Training updates
```

#### Access Control
```yaml
User Roles:
  Admin:
    - Full system access
    - User management
    - System configuration
    
  User:
    - Own resources only
    - API access
    - Dashboard access
    
  Support:
    - Read-only access
    - User assistance
    - Ticket management
```

#### Compliance Requirements
- **GDPR**: EU data protection
- **CCPA**: California privacy
- **YouTube ToS**: Platform compliance
- **PCI DSS**: Payment security
- **COPPA**: Child protection

---

*Document Status: Version 1.0 - January 2025*
*Owner: CTO/Technical Director*
*Review Cycle: Monthly*