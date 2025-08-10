# YTEMPIRE Documentation - Technical Architecture

## 3.1 System Architecture

### High-Level Architecture Overview

```
┌─────────────────────────────────────────────────────────────┐
│                         Users (50-100)                       │
└─────────────────┬───────────────────────────────────────────┘
                  │
                  ▼
┌─────────────────────────────────────────────────────────────┐
│                    Frontend Layer (React)                    │
│  Dashboard │ Channel Mgmt │ Analytics │ Settings │ Wizard    │
└─────────────────┬───────────────────────────────────────────┘
                  │ HTTPS/WebSocket
                  ▼
┌─────────────────────────────────────────────────────────────┐
│                  API Gateway (FastAPI)                       │
│     Auth │ Rate Limiting │ Request Routing │ Logging        │
└─────────────────┬───────────────────────────────────────────┘
                  │
       ┌──────────┴──────────┬──────────┬────────────┐
       ▼                     ▼          ▼            ▼
┌──────────────┐  ┌──────────────┐  ┌──────────┐  ┌──────────┐
│   Backend    │  │     AI/ML    │  │   Queue  │  │   Cache  │
│   Services   │  │   Services   │  │  (Redis) │  │  (Redis) │
└──────┬───────┘  └──────┬───────┘  └──────────┘  └──────────┘
       │                 │
       ▼                 ▼
┌──────────────────────────────────────────────────────────────┐
│              Data Layer (PostgreSQL + Storage)               │
│         Database │ File Storage │ Model Storage              │
└──────────────────────────────────────────────────────────────┘
                         │
                         ▼
┌──────────────────────────────────────────────────────────────┐
│                    External Services                         │
│  YouTube API │ OpenAI │ ElevenLabs │ Stock Media │ Stripe   │
└──────────────────────────────────────────────────────────────┘
```

### Service Boundaries

#### Core Services Architecture

**1. API Gateway Service**
- **Technology**: FastAPI (Python 3.11)
- **Responsibilities**:
  - Request routing and load balancing
  - Authentication and authorization
  - Rate limiting and throttling
  - Request/response transformation
  - API versioning
- **Scaling**: Horizontal scaling with load balancer

**2. User Management Service**
- **Components**:
  - User registration and profiles
  - Authentication (JWT-based)
  - Subscription management
  - Channel ownership tracking
- **Database**: PostgreSQL with user schema

**3. Content Generation Service**
- **Components**:
  - Script generation (GPT integration)
  - Voice synthesis pipeline
  - Video assembly engine
  - Thumbnail generation
- **Processing**: Queue-based async processing

**4. Channel Management Service**
- **Components**:
  - Channel CRUD operations
  - YouTube API integration
  - Publishing scheduler
  - Analytics aggregation
- **Storage**: PostgreSQL + Redis cache

**5. Analytics Service**
- **Components**:
  - Real-time metrics collection
  - Performance aggregation
  - Cost tracking
  - Revenue optimization
- **Database**: PostgreSQL + time-series data

**6. Video Processing Pipeline**
- **Components**:
  - Queue management
  - GPU/CPU scheduling
  - Progress tracking
  - Error recovery
- **Infrastructure**: Celery + Redis

### Data Flow Architecture

#### Video Generation Flow
```
1. User Request → API Gateway
2. API Gateway → Queue (Redis)
3. Queue → AI Service (Trend Analysis)
4. AI Service → Content Generation
5. Content → Voice Synthesis
6. Audio + Visuals → Video Assembly
7. Video → Quality Check
8. Approved Video → YouTube Upload
9. Success → User Notification
```

#### Real-time Updates Flow
```
1. Backend Event → WebSocket Server
2. WebSocket → Frontend Client
3. Frontend → State Update
4. UI → Real-time Display
```

### Database Schema Design

#### Core Tables Structure

**Users Table**
```sql
CREATE TABLE users (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    email VARCHAR(255) UNIQUE NOT NULL,
    password_hash VARCHAR(255) NOT NULL,
    subscription_tier VARCHAR(50),
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    is_active BOOLEAN DEFAULT true,
    metadata JSONB
);
```

**Channels Table**
```sql
CREATE TABLE channels (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    user_id UUID REFERENCES users(id),
    youtube_channel_id VARCHAR(255) UNIQUE,
    channel_name VARCHAR(255),
    niche VARCHAR(100),
    status VARCHAR(50),
    settings JSONB,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);
```

**Videos Table**
```sql
CREATE TABLE videos (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    channel_id UUID REFERENCES channels(id),
    title VARCHAR(500),
    description TEXT,
    youtube_video_id VARCHAR(255),
    status VARCHAR(50),
    generation_cost DECIMAL(10,2),
    processing_time_seconds INTEGER,
    quality_score DECIMAL(3,2),
    metadata JSONB,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    published_at TIMESTAMP
);
```

**Analytics Table**
```sql
CREATE TABLE analytics (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    video_id UUID REFERENCES videos(id),
    views INTEGER DEFAULT 0,
    likes INTEGER DEFAULT 0,
    comments INTEGER DEFAULT 0,
    watch_time_minutes DECIMAL(10,2),
    revenue_cents INTEGER DEFAULT 0,
    ctr DECIMAL(5,2),
    recorded_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);
```

### API Design Patterns

#### RESTful API Structure
```
/api/v1/
├── /auth
│   ├── POST /register
│   ├── POST /login
│   ├── POST /refresh
│   └── POST /logout
├── /users
│   ├── GET /profile
│   ├── PUT /profile
│   └── DELETE /account
├── /channels
│   ├── GET /
│   ├── POST /
│   ├── GET /{id}
│   ├── PUT /{id}
│   └── DELETE /{id}
├── /videos
│   ├── GET /
│   ├── POST /generate
│   ├── GET /{id}
│   ├── GET /{id}/status
│   └── DELETE /{id}
├── /analytics
│   ├── GET /dashboard
│   ├── GET /channels/{id}/metrics
│   └── GET /videos/{id}/performance
└── /webhooks
    ├── POST /youtube
    ├── POST /stripe
    └── POST /n8n
```

#### WebSocket Events
```javascript
// Client → Server Events
{
  "subscribe_channel_updates": { "channel_id": "uuid" },
  "subscribe_video_progress": { "video_id": "uuid" },
  "subscribe_analytics": { "channel_ids": ["uuid1", "uuid2"] }
}

// Server → Client Events
{
  "video_progress": { 
    "video_id": "uuid",
    "status": "processing",
    "progress": 45,
    "stage": "voice_synthesis"
  },
  "channel_update": {
    "channel_id": "uuid",
    "event": "video_published",
    "data": { /* video details */ }
  },
  "analytics_update": {
    "channel_id": "uuid",
    "metrics": { /* real-time metrics */ }
  }
}
```

## 3.2 Technology Stack

### Core Technology Decisions

#### Backend Stack
| Component | Technology | Version | Justification |
|-----------|------------|---------|---------------|
| **Language** | Python | 3.11 | Async support, ML ecosystem, team expertise |
| **Framework** | FastAPI | 0.104+ | Modern, async, automatic OpenAPI docs |
| **Database** | PostgreSQL | 15 | ACID compliance, JSON support, proven scale |
| **Cache** | Redis | 7 | Sub-ms latency, pub/sub, queue support |
| **Queue** | Celery | 5.3 | Distributed task processing, monitoring |
| **ORM** | SQLAlchemy | 2.0 | Async support, powerful queries |

#### Frontend Stack
| Component | Technology | Version | Justification |
|-----------|------------|---------|---------------|
| **Framework** | React | 18.2 | Proven, large ecosystem, team knowledge |
| **Language** | TypeScript | 5.3 | Type safety, better IDE support |
| **Build Tool** | Vite | 5.0 | Fast HMR, optimized builds |
| **State Mgmt** | Zustand | 4.4 | Simple API, TypeScript-first, lightweight |
| **UI Library** | Material-UI | 5.14 | Comprehensive components, theming |
| **Charts** | Recharts | 2.10 | React-native, good performance |
| **Forms** | React Hook Form | 7.x | Performance, validation |
| **Testing** | Vitest | 1.0 | Fast, Vite integration |

#### AI/ML Stack
| Component | Technology | Version | Justification |
|-----------|------------|---------|---------------|
| **Deep Learning** | PyTorch | 2.0 | Flexibility, GPU support |
| **NLP** | Transformers | 4.35 | State-of-art models |
| **Time Series** | Prophet | 1.1 | Facebook's forecasting |
| **LLM** | OpenAI GPT | 4/3.5 | Best quality/cost ratio |
| **Voice** | ElevenLabs | API | Most natural voices |
| **Images** | Stable Diffusion | XL | Open source, quality |

#### Infrastructure Stack
| Component | Technology | Version | Justification |
|-----------|------------|---------|---------------|
| **OS** | Ubuntu | 22.04 LTS | Stability, Docker support |
| **Container** | Docker | 24.x | Industry standard |
| **Orchestration** | Docker Compose | 2.x | Simple for MVP |
| **Proxy** | Nginx | 1.24 | Performance, features |
| **Monitoring** | Prometheus | 2.47 | Metrics collection |
| **Visualization** | Grafana | 10.x | Beautiful dashboards |
| **Automation** | N8N | Latest | Visual workflows |

### Development Tools

#### Code Quality & Testing
- **Linting**: ESLint, Pylint, Black
- **Type Checking**: TypeScript, mypy
- **Unit Testing**: Jest, Pytest
- **Integration Testing**: Supertest, pytest-asyncio
- **E2E Testing**: Playwright
- **Load Testing**: k6, Locust

#### CI/CD Pipeline
- **Version Control**: Git + GitHub
- **CI/CD**: GitHub Actions
- **Code Review**: GitHub Pull Requests
- **Artifact Storage**: GitHub Packages
- **Deployment**: Docker + custom scripts

#### Monitoring & Observability
- **Metrics**: Prometheus + Grafana
- **Logging**: Docker logs + logrotate
- **Tracing**: OpenTelemetry (future)
- **Alerting**: AlertManager + Slack
- **Error Tracking**: Sentry (future)

## 3.3 Infrastructure Design

### MVP Infrastructure (Weeks 1-12)

#### Local Server Specifications
```yaml
Hardware:
  CPU: AMD Ryzen 9 9950X3D (16 cores, 32 threads)
  RAM: 128GB DDR5-6000
  GPU: NVIDIA RTX 5090 (32GB VRAM)
  Storage:
    - System: 2TB NVMe Gen5 SSD
    - Data: 8TB NVMe Gen4 SSD
    - Backup: 8TB External HDD
  Network: 1Gbps Fiber (symmetric)
  UPS: 1500VA battery backup

Operating System:
  OS: Ubuntu 22.04 LTS Server
  Kernel: Optimized for containers
  Drivers: NVIDIA CUDA 12.x

Resource Allocation:
  CPU Distribution:
    - PostgreSQL: 4 cores
    - Backend Services: 4 cores
    - N8N Automation: 2 cores
    - Frontend: 2 cores
    - Monitoring: 2 cores
    - System/Reserve: 2 cores
  
  Memory Distribution:
    - PostgreSQL: 16GB
    - Redis: 8GB
    - Backend Services: 24GB
    - N8N: 8GB
    - Frontend: 8GB
    - Video Processing: 48GB
    - System/Cache: 16GB
  
  Storage Distribution:
    - System/OS: 200GB
    - Database: 300GB
    - Applications: 500GB
    - Videos/Media: 6TB
    - Backups: 1TB
    - Logs/Temp: 1TB
```

#### Container Architecture
```yaml
Docker Compose Services:
  # Core Services
  api:
    image: ytempire-api:latest
    ports: ["8080:8080"]
    volumes: ["./data:/data"]
    environment:
      - DATABASE_URL
      - REDIS_URL
      - JWT_SECRET
    
  frontend:
    image: ytempire-frontend:latest
    ports: ["3000:80"]
    volumes: ["./static:/static"]
  
  # Databases
  postgres:
    image: postgres:15-alpine
    ports: ["5432:5432"]
    volumes: ["postgres-data:/var/lib/postgresql/data"]
    environment:
      - POSTGRES_PASSWORD
  
  redis:
    image: redis:7-alpine
    ports: ["6379:6379"]
    volumes: ["redis-data:/data"]
    command: redis-server --appendonly yes
  
  # Processing
  worker:
    image: ytempire-worker:latest
    volumes: ["./videos:/videos"]
    environment:
      - GPU_DEVICE=/dev/nvidia0
  
  n8n:
    image: n8nio/n8n:latest
    ports: ["5678:5678"]
    volumes: ["n8n-data:/home/node/.n8n"]
  
  # Monitoring
  prometheus:
    image: prom/prometheus:latest
    ports: ["9090:9090"]
    volumes: ["./prometheus.yml:/etc/prometheus/prometheus.yml"]
  
  grafana:
    image: grafana/grafana:latest
    ports: ["3001:3000"]
    volumes: ["grafana-data:/var/lib/grafana"]
```

### Scale Infrastructure (Post-MVP)

#### Cloud Migration Path (Months 4-6)

**Phase 1: Hybrid Approach**
- Keep video processing on local GPU
- Move web services to cloud
- Cloud database with local replica
- CDN for static assets

**Phase 2: Full Cloud (Month 6+)**
```yaml
GCP Architecture:
  Compute:
    - GKE Cluster: n2-standard-8 nodes
    - GPU Pool: nvidia-tesla-t4 instances
    - Autoscaling: 3-20 nodes
  
  Storage:
    - Cloud SQL: PostgreSQL HA
    - Memorystore: Redis cluster
    - Cloud Storage: Video assets
    - CDN: Global distribution
  
  Networking:
    - Load Balancer: Global HTTPS
    - Cloud Armor: DDoS protection
    - Private VPC: Internal services
    - Cloud NAT: Outbound traffic
```

### Scaling Strategy

#### Horizontal Scaling Plan
```
Current (MVP):
- 1 server
- 50 users
- 250 channels
- 50 videos/day

Month 3:
- 1 server + cloud services
- 100 users
- 500 channels
- 150 videos/day

Month 6:
- Cloud infrastructure
- 200 users
- 1000 channels
- 500 videos/day

Year 1:
- Multi-region cloud
- 1000+ users
- 5000+ channels
- 5000+ videos/day
```

## 3.4 Security Architecture

### Authentication & Authorization

#### JWT-Based Authentication
```python
# Token Structure
{
  "header": {
    "alg": "HS256",
    "typ": "JWT"
  },
  "payload": {
    "sub": "user_uuid",
    "email": "user@example.com",
    "subscription": "professional",
    "exp": 1234567890,
    "iat": 1234567890,
    "roles": ["user"],
    "permissions": ["channel:create", "video:generate"]
  }
}
```

#### Authorization Model
```yaml
Roles:
  admin:
    - Full system access
    - User management
    - System configuration
  
  user:
    - Own channels management
    - Video generation
    - Analytics viewing
  
  beta_user:
    - Extended limits
    - Early feature access
    - Priority support

Permissions:
  channel:
    - create (max 5 for starter)
    - read (own channels)
    - update (own channels)
    - delete (own channels)
  
  video:
    - generate (rate limited)
    - read (own videos)
    - delete (own videos)
  
  analytics:
    - read (own data)
    - export (subscription based)
```

### Security Measures

#### Application Security
- **Input Validation**: All inputs sanitized
- **SQL Injection Prevention**: Parameterized queries
- **XSS Protection**: Content Security Policy
- **CSRF Protection**: Token validation
- **Rate Limiting**: Per-user and per-IP

#### Infrastructure Security
- **Network Security**:
  - UFW firewall configuration
  - Fail2ban for intrusion prevention
  - SSH key-only authentication
  - VPN for admin access

- **Data Security**:
  - Encryption at rest (AES-256)
  - Encryption in transit (TLS 1.3)
  - Secure secret management
  - Regular security updates

- **Monitoring**:
  - Failed login tracking
  - Unusual activity detection
  - Security event logging
  - Regular vulnerability scanning

### Compliance & Privacy

#### Data Protection
- **GDPR Compliance**:
  - User consent management
  - Data export capability
  - Right to deletion
  - Privacy policy

- **YouTube Compliance**:
  - API quota management
  - Terms of service adherence
  - Content policy compliance
  - Copyright protection

#### Audit & Logging
```yaml
Audit Events:
  - User login/logout
  - Channel creation/deletion
  - Video generation requests
  - Payment transactions
  - Configuration changes
  - Admin actions

Log Retention:
  - Security logs: 90 days
  - Application logs: 30 days
  - Access logs: 30 days
  - Audit logs: 1 year
```

### Disaster Recovery

#### Backup Strategy
- **Database**: Daily automated backups, 30-day retention
- **Files**: Incremental daily, weekly full backups
- **Configuration**: Version controlled in Git
- **Recovery Time Objective (RTO)**: 4 hours
- **Recovery Point Objective (RPO)**: 24 hours

#### Incident Response Plan
1. **Detection**: Automated monitoring alerts
2. **Assessment**: Severity classification
3. **Containment**: Isolate affected systems
4. **Eradication**: Remove threat
5. **Recovery**: Restore from backups
6. **Lessons Learned**: Post-mortem analysis

---

*Document Version: 1.0*  
*Last Updated: January 2025*  
*Status: FINAL - Architecture Approved*  
*Owner: CTO/Technical Director*