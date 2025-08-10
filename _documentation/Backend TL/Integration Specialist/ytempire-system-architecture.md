# YTEMPIRE System Architecture

## 2.1 High-Level Architecture

### System Overview

```
┌─────────────────────────────────────────────────────────────┐
│                         Users (50 Beta)                      │
└────────────────────────────┬─────────────────────────────────┘
                             │ HTTPS
                             ▼
┌─────────────────────────────────────────────────────────────┐
│                     Frontend (React/Nginx)                   │
│                   Dashboard, Analytics, Controls             │
└────────────────────────────┬─────────────────────────────────┘
                             │ REST API / WebSocket
                             ▼
┌─────────────────────────────────────────────────────────────┐
│                    API Gateway (FastAPI)                     │
│              Authentication, Rate Limiting, Routing          │
└────────────────────────────┬─────────────────────────────────┘
                             │
        ┌────────────────────┼────────────────────┐
        ▼                    ▼                    ▼
┌──────────────┐    ┌──────────────┐    ┌──────────────┐
│   Backend    │    │   N8N        │    │   AI/ML      │
│   Services   │◄──►│   Workflows  │◄──►│   Services   │
└──────────────┘    └──────────────┘    └──────────────┘
        │                    │                    │
        └────────────────────┼────────────────────┘
                             ▼
┌─────────────────────────────────────────────────────────────┐
│                    Data Layer                                │
│         PostgreSQL (Primary) | Redis (Cache/Queue)           │
└─────────────────────────────────────────────────────────────┘
                             │
                             ▼
┌─────────────────────────────────────────────────────────────┐
│                   External Services                          │
│   YouTube APIs | OpenAI | Google TTS | Stripe | Stock Media │
└─────────────────────────────────────────────────────────────┘
```

### Component Architecture

#### Core Services Architecture
```yaml
Application_Layer:
  Frontend:
    Technology: React 18 + TypeScript
    State: Zustand
    UI: Material-UI
    Charts: Recharts
    Hosting: Nginx container
    
  API_Gateway:
    Framework: FastAPI
    Authentication: JWT
    Rate_Limiting: Redis-based
    Documentation: OpenAPI/Swagger
    
  Backend_Services:
    - User_Management
    - Channel_Management
    - Video_Processing
    - Analytics_Service
    - Cost_Tracking
    - Payment_Processing
    
  Workflow_Engine:
    Platform: N8N
    Execution: Docker container
    Storage: PostgreSQL
    Queue: Redis
    
  AI_ML_Services:
    - Trend_Prediction
    - Content_Generation
    - Quality_Scoring
    - Optimization_Engine
```

### Data Flow Architecture

```yaml
Content_Generation_Flow:
  1_Trigger:
    Source: [Schedule, Manual, API]
    Handler: N8N Webhook
    
  2_Processing:
    - Trend_Analysis → AI Service
    - Script_Generation → OpenAI API
    - Voice_Synthesis → Google TTS
    - Video_Assembly → Local GPU
    - Thumbnail_Creation → AI Generator
    
  3_Upload:
    - YouTube_Account_Selection
    - Quota_Check
    - Upload_Execution
    - Metadata_Update
    
  4_Monitoring:
    - Performance_Tracking
    - Cost_Calculation
    - Analytics_Update
    - User_Notification
```

## 2.2 Technical Stack Decisions

### Core Technology Stack

#### Backend Stack
```yaml
Language: Python 3.11+
Framework: FastAPI
Async: asyncio + aiohttp
ORM: SQLAlchemy 2.0
Validation: Pydantic
Testing: Pytest + pytest-asyncio
Documentation: OpenAPI/Swagger
```

#### Frontend Stack
```yaml
Framework: React 18.2
Language: TypeScript 5.3
Build: Vite 5.0
State: Zustand 4.4
UI: Material-UI 5.14
Charts: Recharts 2.10
Forms: React Hook Form 7.x
Testing: Vitest + React Testing Library
```

#### Infrastructure Stack
```yaml
OS: Ubuntu 22.04 LTS
Containers: Docker 24.x + Docker Compose 2.x
Reverse_Proxy: Nginx
SSL: Let's Encrypt
Monitoring: Prometheus + Grafana
Logs: Docker logs + logrotate
Backup: rsync + cloud storage
```

#### Data Stack
```yaml
Primary_DB: PostgreSQL 15
Cache: Redis 7
Queue: Celery + Redis
Search: PostgreSQL Full Text
Analytics: TimescaleDB extension
Backup: pg_dump + S3 (future)
```

### Technology Justifications

#### Why FastAPI over Django/Flask?
- **Async Support**: Native async/await for high concurrency
- **Performance**: 3x faster than Flask, on par with Node.js
- **Auto Documentation**: OpenAPI/Swagger built-in
- **Type Safety**: Pydantic validation
- **Modern**: Built for modern Python practices

#### Why Zustand over Redux?
- **Simplicity**: 90% less boilerplate code
- **Size**: 8KB vs Redux's 50KB
- **Performance**: Direct state updates
- **Learning Curve**: Team can be productive in hours
- **TypeScript**: First-class TypeScript support

#### Why PostgreSQL over MongoDB?
- **ACID Compliance**: Critical for payment/user data
- **Relationships**: Complex relationships between entities
- **JSON Support**: JSONB for flexible data
- **Maturity**: Battle-tested at scale
- **Extensions**: TimescaleDB, PostGIS available

#### Why N8N over Airflow/Temporal?
- **Visual Workflows**: Non-developers can understand
- **Rapid Development**: 10x faster workflow creation
- **Built-in Integrations**: 200+ pre-built nodes
- **Self-Hosted**: Full control and customization
- **Cost**: Free for self-hosted

#### Why Local Server over Cloud (MVP)?
- **Cost**: $300/month vs $3,000/month cloud
- **Control**: Complete control over resources
- **Performance**: No network latency
- **Security**: Data never leaves premises
- **Simplicity**: Reduced complexity for MVP

## 2.3 Database Design

### Database Schema Overview

```sql
-- Database: ytempire_mvp
-- Total Schemas: 6 main schemas

CREATE DATABASE ytempire_mvp;

-- Schemas
CREATE SCHEMA users;      -- User management and authentication
CREATE SCHEMA channels;   -- YouTube channel management
CREATE SCHEMA videos;     -- Video generation and tracking
CREATE SCHEMA costs;      -- Cost tracking and optimization
CREATE SCHEMA payments;   -- Subscription and payment data
CREATE SCHEMA system;     -- System monitoring and audit
```

### Core Schema Design

#### Users Schema
```sql
-- users.accounts: Core user table
CREATE TABLE users.accounts (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    email VARCHAR(255) UNIQUE NOT NULL,
    email_verified BOOLEAN DEFAULT FALSE,
    password_hash VARCHAR(255) NOT NULL,
    full_name VARCHAR(255),
    
    -- Subscription
    stripe_customer_id VARCHAR(255) UNIQUE,
    subscription_tier VARCHAR(50) DEFAULT 'free',
    subscription_status VARCHAR(50) DEFAULT 'inactive',
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

-- users.sessions: JWT session tracking
CREATE TABLE users.sessions (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    user_id UUID NOT NULL REFERENCES users.accounts(id) ON DELETE CASCADE,
    token_hash VARCHAR(255) UNIQUE NOT NULL,
    refresh_token_hash VARCHAR(255) UNIQUE,
    expires_at TIMESTAMP WITH TIME ZONE NOT NULL,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);
```

#### Channels Schema
```sql
-- channels.youtube_channels: YouTube channel management
CREATE TABLE channels.youtube_channels (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    user_id UUID NOT NULL REFERENCES users.accounts(id),
    
    -- YouTube Info
    youtube_channel_id VARCHAR(255) UNIQUE NOT NULL,
    channel_title VARCHAR(255) NOT NULL,
    channel_description TEXT,
    
    -- Configuration
    niche VARCHAR(100) NOT NULL,
    automation_enabled BOOLEAN DEFAULT TRUE,
    videos_per_day INTEGER DEFAULT 3,
    
    -- Metrics
    subscriber_count INTEGER DEFAULT 0,
    total_views BIGINT DEFAULT 0,
    monthly_revenue DECIMAL(10,2) DEFAULT 0.00,
    health_score DECIMAL(3,2) DEFAULT 1.00,
    
    -- OAuth
    oauth_credentials JSONB, -- Encrypted
    oauth_expires_at TIMESTAMP WITH TIME ZONE,
    
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);
```

#### Videos Schema
```sql
-- videos.video_records: Video generation tracking
CREATE TABLE videos.video_records (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    channel_id UUID NOT NULL REFERENCES channels.youtube_channels(id),
    user_id UUID NOT NULL REFERENCES users.accounts(id),
    
    -- Content
    title VARCHAR(500) NOT NULL,
    description TEXT,
    script TEXT,
    tags TEXT[],
    
    -- Files
    video_file_path VARCHAR(500),
    thumbnail_file_path VARCHAR(500),
    
    -- YouTube
    youtube_video_id VARCHAR(255) UNIQUE,
    youtube_url VARCHAR(500),
    status VARCHAR(50) DEFAULT 'queued',
    
    -- Metrics
    processing_duration_seconds INTEGER,
    quality_score DECIMAL(3,2),
    
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);
```

#### Costs Schema
```sql
-- costs.video_costs: Detailed cost tracking
CREATE TABLE costs.video_costs (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    video_id UUID NOT NULL REFERENCES videos.video_records(id),
    
    -- Service Costs
    script_generation_cost DECIMAL(10,4) DEFAULT 0.0000,
    voice_synthesis_cost DECIMAL(10,4) DEFAULT 0.0000,
    video_processing_cost DECIMAL(10,4) DEFAULT 0.0000,
    total_cost DECIMAL(10,4) DEFAULT 0.0000,
    
    -- Optimization Level
    optimization_level VARCHAR(20) DEFAULT 'standard',
    
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);
```

### Database Performance Optimizations

```sql
-- Critical Indexes
CREATE INDEX idx_videos_status ON videos.video_records(status);
CREATE INDEX idx_videos_created ON videos.video_records(created_at DESC);
CREATE INDEX idx_channels_user ON channels.youtube_channels(user_id);
CREATE INDEX idx_costs_video ON costs.video_costs(video_id);

-- Partitioning for scale (future)
-- Partition videos table by month
-- Partition costs table by month
-- Archive old data to cold storage
```

## 2.4 Infrastructure Specifications

### Hardware Specifications (MVP - Local Server)

```yaml
Server_Configuration:
  CPU:
    Model: AMD Ryzen 9 9950X3D
    Cores: 16 cores (32 threads)
    Clock: 4.3 GHz base, 5.7 GHz boost
    
  Memory:
    Capacity: 128GB DDR5
    Speed: 5600 MHz
    Configuration: 4x32GB
    
  GPU:
    Model: NVIDIA RTX 5090
    VRAM: 32GB GDDR7
    CUDA_Cores: 21,760
    Purpose: Video processing, AI inference
    
  Storage:
    System: 2TB NVMe SSD (Gen5)
    Data: 4TB NVMe SSD (Gen4)
    Backup: 8TB HDD (RAID 1)
    
  Network:
    Connection: 1Gbps Fiber
    Backup: 100Mbps Cable
    Router: Enterprise-grade
```

### Container Architecture

```yaml
Docker_Services:
  Frontend:
    Image: node:18-alpine
    Memory: 2GB
    CPU: 2 cores
    Ports: 3000
    
  API_Gateway:
    Image: python:3.11-slim
    Memory: 4GB
    CPU: 4 cores
    Ports: 8000
    
  PostgreSQL:
    Image: postgres:15
    Memory: 16GB
    CPU: 4 cores
    Storage: 300GB
    
  Redis:
    Image: redis:7-alpine
    Memory: 8GB
    CPU: 2 cores
    Persistence: AOF
    
  N8N:
    Image: n8nio/n8n:latest
    Memory: 4GB
    CPU: 2 cores
    Storage: 50GB
    
  Monitoring:
    Prometheus: 2GB RAM, 1 core
    Grafana: 1GB RAM, 1 core
```

### Network Architecture

```yaml
Network_Topology:
  External:
    Domain: ytempire.com
    SSL: Let's Encrypt
    CDN: CloudFlare (future)
    
  Internal:
    Docker_Network: Bridge mode
    Service_Discovery: Docker DNS
    Load_Balancing: Nginx
    
  Security:
    Firewall: UFW
    Ports_Open: [80, 443, 22]
    VPN: WireGuard (admin access)
```

### Scaling Infrastructure (Post-MVP)

```yaml
Phase_2_Hybrid (Months 4-6):
  Local:
    - Core services
    - Primary database
    - Video processing
  Cloud:
    - S3 backup storage
    - CloudFlare CDN
    - Disaster recovery
    
Phase_3_Cloud (Months 7-12):
  AWS_Services:
    - ECS for containers
    - RDS for PostgreSQL
    - ElastiCache for Redis
    - S3 for storage
    - CloudFront CDN
    - Auto-scaling groups
```

## 2.5 Security Architecture

### Security Layers

```yaml
Application_Security:
  Authentication:
    Method: JWT with refresh tokens
    Expiry: 15 minutes (access), 7 days (refresh)
    Storage: HttpOnly cookies
    
  Authorization:
    Model: RBAC (Role-Based Access Control)
    Roles: [admin, user, beta_user]
    Permissions: Granular per resource
    
  Password_Security:
    Hashing: bcrypt (cost factor 12)
    Requirements: 8+ chars, mixed case, numbers
    Reset: Email-based with expiring tokens
```

### API Security

```yaml
API_Protection:
  Rate_Limiting:
    Anonymous: 60 requests/minute
    Authenticated: 600 requests/minute
    Implementation: Redis-based
    
  Input_Validation:
    Framework: Pydantic
    SQL_Injection: Parameterized queries
    XSS: Content sanitization
    
  CORS:
    Origins: ['https://ytempire.com']
    Methods: ['GET', 'POST', 'PUT', 'DELETE']
    Credentials: true
```

### Infrastructure Security

```yaml
Server_Security:
  OS_Hardening:
    - Disable root SSH
    - Key-based authentication only
    - Automatic security updates
    - Minimal installed packages
    
  Network_Security:
    Firewall: UFW with strict rules
    DDoS: CloudFlare (future)
    IDS: Fail2ban
    VPN: WireGuard for admin
    
  Data_Security:
    Encryption_at_Rest: LUKS
    Encryption_in_Transit: TLS 1.3
    Backup_Encryption: GPG
    Secrets_Management: Environment variables
```

### Compliance & Privacy

```yaml
Data_Protection:
  GDPR_Compliance:
    - User consent mechanisms
    - Data export capability
    - Right to deletion
    - Privacy policy
    
  PCI_Compliance:
    - No credit card storage
    - Stripe handles all payments
    - Tokenization only
    
  YouTube_Compliance:
    - API terms adherence
    - Content policy checks
    - Copyright screening
```

### Security Monitoring

```yaml
Monitoring_Strategy:
  Real_time:
    - Failed login attempts
    - API anomalies
    - Resource spikes
    - Error rates
    
  Alerts:
    Critical:
      - Multiple failed logins
      - Unauthorized access attempts
      - Service outages
    Warning:
      - High error rates
      - Resource exhaustion
      - Certificate expiry
      
  Audit_Logging:
    - All API calls
    - Authentication events
    - Configuration changes
    - Data access patterns
```

### Incident Response Plan

```yaml
Incident_Response:
  Detection:
    - Automated monitoring
    - User reports
    - Security scans
    
  Response_Team:
    Primary: Security Engineer
    Secondary: Platform Ops Lead
    Escalation: CTO
    
  Procedures:
    1. Isolate affected systems
    2. Assess damage scope
    3. Preserve evidence
    4. Remediate vulnerability
    5. Restore services
    6. Post-mortem analysis
    
  Communication:
    Internal: Slack #security-incidents
    External: Status page updates
    Legal: As required
```