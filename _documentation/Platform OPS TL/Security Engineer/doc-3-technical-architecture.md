# 3. TECHNICAL ARCHITECTURE

## 3.1 Infrastructure Architecture

### MVP Local Deployment

#### Physical Infrastructure
```yaml
deployment_model: Local Server (On-Premises)
location: Single physical location
redundancy: External backup drives only
network: 1Gbps fiber connection
power: UPS backup (4 hours)
```

#### Network Architecture
```
Internet (1Gbps Fiber)
    │
    ├── Firewall/Router
    │   ├── DMZ (10.0.40.0/24)
    │   │   └── Nginx Reverse Proxy
    │   │
    │   ├── App Network (10.0.20.0/24)
    │   │   ├── Backend Services
    │   │   ├── Frontend Server
    │   │   └── N8N Automation
    │   │
    │   ├── Data Network (10.0.30.0/24)
    │   │   ├── PostgreSQL
    │   │   └── Redis
    │   │
    │   └── Management (10.0.10.0/24)
    │       ├── Monitoring Stack
    │       └── Admin Access
```

### Hardware Specifications

#### Server Configuration
```yaml
Server Specifications:
  Model: Custom Build
  CPU: AMD Ryzen 9 9950X3D (16 cores, 32 threads)
  RAM: 128GB DDR5-6000
  GPU: NVIDIA RTX 5090 (32GB VRAM)
  
  Storage:
    System: 2TB NVMe SSD (OS & Applications)
    Data: 4TB NVMe SSD (Database & Cache)
    Media: 8TB NVMe SSD (Videos & Assets)
    Backup: 8TB External USB 3.2 Drives (2x)
  
  Network:
    Primary: 10Gb Ethernet to switch
    External: 1Gbps fiber to ISP
    Internal: Gigabit managed switch
  
  Total Cost: $10,000 (already allocated)
```

#### Resource Allocation
```yaml
CPU Allocation (16 cores):
  PostgreSQL: 4 cores
  Backend Services: 4 cores
  Video Processing: 4 cores
  Frontend/N8N: 2 cores
  System/Monitoring: 2 cores

Memory Allocation (128GB):
  PostgreSQL: 16GB
  Redis: 8GB
  Backend Services: 24GB
  Video Processing: 48GB
  Frontend: 8GB
  N8N: 8GB
  System/Buffer: 16GB

GPU Allocation:
  Video Rendering: 80%
  Thumbnail Generation: 15%
  ML Inference: 5%
```

### Software Stack

#### Base System
```yaml
Operating System:
  OS: Ubuntu 22.04 LTS
  Kernel: 5.15 LTS (optimized for containers)
  Filesystem: ext4 with LVM
  
Container Platform:
  Runtime: Docker 24.0+
  Orchestration: Docker Compose v2.20+
  Registry: Local registry (Harbor)
  
Networking:
  Firewall: UFW + iptables
  Proxy: Nginx 1.24+
  SSL: Let's Encrypt (Certbot)
  VPN: WireGuard (admin access)
```

#### Service Stack
```yaml
Application Services:
  Backend: FastAPI (Python 3.11)
  Frontend: React 18 + Vite
  Automation: N8N (latest)
  Queue: Celery + Redis
  
Data Services:
  Database: PostgreSQL 15
  Cache: Redis 7
  Search: PostgreSQL FTS
  File Storage: Local filesystem
  
Monitoring:
  Metrics: Prometheus
  Visualization: Grafana
  Logs: Docker logs + Loki
  Alerts: Alertmanager
  
Security:
  WAF: ModSecurity (Nginx)
  IDS: Fail2ban
  Secrets: Environment files
  Backup: Restic
```

### Future Cloud Migration Path

#### Phase 1: Hybrid Model (Months 4-6)
```yaml
Hybrid Architecture:
  Local:
    - Video processing (GPU intensive)
    - Primary database
    - Real-time services
  
  Cloud:
    - Static assets (CDN)
    - Backup storage
    - Disaster recovery
    - Monitoring
```

#### Phase 2: Cloud-First (Year 2)
```yaml
Target Architecture:
  Provider: AWS or GCP
  
  Compute:
    - EKS/GKE for orchestration
    - GPU instances for processing
    - Auto-scaling groups
  
  Storage:
    - RDS for PostgreSQL
    - ElastiCache for Redis
    - S3/GCS for media
  
  Network:
    - CloudFront/Cloud CDN
    - Load balancers
    - VPC with multiple AZs
```

---

## 3.2 Application Architecture

### Service Architecture

#### Microservices Design
```yaml
Services Map:
  API Gateway:
    Port: 8000
    Framework: Kong/Nginx
    Responsibilities:
      - Rate limiting
      - Authentication
      - Request routing
      - SSL termination
  
  Main API:
    Port: 8001
    Framework: FastAPI
    Database: PostgreSQL
    Cache: Redis
    Responsibilities:
      - User management
      - Channel operations
      - Video metadata
      - Analytics
  
  Video Processor:
    Port: 8002
    Framework: Python + FFmpeg
    Queue: Celery
    GPU: Required
    Responsibilities:
      - Video compilation
      - Rendering
      - Thumbnail generation
      - Upload to YouTube
  
  Analytics Service:
    Port: 8003
    Framework: FastAPI
    Database: PostgreSQL
    Responsibilities:
      - YouTube Analytics sync
      - Performance metrics
      - Cost calculations
      - Revenue tracking
  
  Webhook Handler:
    Port: 8004
    Framework: FastAPI
    Queue: Redis
    Responsibilities:
      - YouTube webhooks
      - Stripe webhooks
      - Event processing
```

### API Specifications

#### Authentication & Authorization
```python
# JWT Configuration
JWT_CONFIG = {
    'algorithm': 'RS256',
    'access_token_expire': 3600,  # 1 hour
    'refresh_token_expire': 604800,  # 7 days
    'issuer': 'ytempire.com',
    'audience': ['api.ytempire.com']
}

# RBAC Roles
ROLES = {
    'admin': ['all'],
    'user': ['channels:*', 'videos:*', 'analytics:read'],
    'viewer': ['channels:read', 'videos:read'],
    'service': ['internal:*']
}
```

#### Core API Endpoints
```yaml
User Management:
  POST /auth/register
  POST /auth/login
  POST /auth/refresh
  GET /users/profile
  PUT /users/profile

Channel Management:
  GET /channels
  POST /channels
  GET /channels/{id}
  PUT /channels/{id}
  DELETE /channels/{id}

Video Operations:
  GET /videos
  POST /videos/generate
  GET /videos/{id}
  PUT /videos/{id}
  DELETE /videos/{id}
  POST /videos/{id}/publish

Analytics:
  GET /analytics/overview
  GET /analytics/channels/{id}
  GET /analytics/videos/{id}
  GET /analytics/revenue
  GET /analytics/costs
```

### Database Design

#### Schema Overview
```sql
-- Core Tables
users (
  id UUID PRIMARY KEY,
  email VARCHAR(255) UNIQUE,
  password_hash VARCHAR(255),
  subscription_status VARCHAR(50),
  created_at TIMESTAMP,
  updated_at TIMESTAMP
)

channels (
  id UUID PRIMARY KEY,
  user_id UUID REFERENCES users(id),
  youtube_channel_id VARCHAR(255),
  name VARCHAR(255),
  description TEXT,
  settings JSONB,
  created_at TIMESTAMP
)

videos (
  id UUID PRIMARY KEY,
  channel_id UUID REFERENCES channels(id),
  title VARCHAR(255),
  description TEXT,
  status VARCHAR(50),
  youtube_video_id VARCHAR(255),
  cost_breakdown JSONB,
  created_at TIMESTAMP,
  published_at TIMESTAMP
)

-- Analytics Tables
video_analytics (
  id UUID PRIMARY KEY,
  video_id UUID REFERENCES videos(id),
  views INTEGER,
  likes INTEGER,
  comments INTEGER,
  revenue DECIMAL(10,2),
  timestamp TIMESTAMP
)
```

### Integration Points

#### External API Integrations
```yaml
OpenAI GPT-4:
  Purpose: Script generation
  Cost: ~$0.50/video
  Rate Limit: 10,000 tokens/min
  Fallback: GPT-3.5 Turbo

ElevenLabs:
  Purpose: Voice synthesis
  Cost: ~$0.30/video
  Rate Limit: 100 requests/min
  Fallback: Google TTS

YouTube API v3:
  Purpose: Publishing, analytics
  Quota: 10,000 units/day
  Critical Endpoints:
    - videos.insert
    - channels.list
    - analytics.reports.query

Stripe:
  Purpose: Payment processing
  Fees: 2.9% + $0.30
  Webhooks:
    - payment_intent.succeeded
    - subscription.updated
    - invoice.paid
```

---

## 3.3 Security Architecture

### Security Principles

#### Core Security Tenets
1. **Defense in Depth**: Multiple layers of security controls
2. **Least Privilege**: Minimal access required for function
3. **Zero Trust**: Verify everything, trust nothing
4. **Secure by Default**: Security built-in, not bolted-on
5. **Continuous Monitoring**: Real-time threat detection

### Authentication & Authorization

#### Implementation Details
```python
# Authentication Flow
class AuthenticationSystem:
    def __init__(self):
        self.jwt_private_key = load_rsa_key('private.pem')
        self.jwt_public_key = load_rsa_key('public.pem')
    
    def generate_tokens(self, user_id: str):
        access_token = create_jwt(
            user_id=user_id,
            expires=3600,
            type='access'
        )
        refresh_token = create_jwt(
            user_id=user_id,
            expires=604800,
            type='refresh'
        )
        return access_token, refresh_token
    
    def verify_token(self, token: str):
        return verify_jwt(token, self.jwt_public_key)
```

#### Session Management
```yaml
Session Configuration:
  Storage: Redis
  TTL: 24 hours
  Refresh: Sliding window
  Max Sessions: 5 per user
  
Security Features:
  - Session fixation protection
  - CSRF tokens
  - Secure cookie flags
  - SameSite attribute
```

### Network Security

#### Firewall Rules
```bash
# UFW Configuration
ufw default deny incoming
ufw default allow outgoing

# Allow SSH (rate limited)
ufw limit 22/tcp

# Allow HTTP/HTTPS
ufw allow 80/tcp
ufw allow 443/tcp

# Internal services (from app network only)
ufw allow from 10.0.20.0/24 to any port 5432
ufw allow from 10.0.20.0/24 to any port 6379

# Monitoring (from management network)
ufw allow from 10.0.10.0/24 to any port 9090
ufw allow from 10.0.10.0/24 to any port 3000
```

#### SSL/TLS Configuration
```nginx
# Nginx SSL Configuration
ssl_certificate /etc/letsencrypt/live/ytempire.com/fullchain.pem;
ssl_certificate_key /etc/letsencrypt/live/ytempire.com/privkey.pem;

ssl_protocols TLSv1.2 TLSv1.3;
ssl_ciphers ECDHE-RSA-AES128-GCM-SHA256:ECDHE-RSA-AES256-GCM-SHA384;
ssl_prefer_server_ciphers off;

ssl_session_timeout 1d;
ssl_session_cache shared:SSL:10m;
ssl_stapling on;
ssl_stapling_verify on;

add_header Strict-Transport-Security "max-age=63072000" always;
```

### Data Protection

#### Encryption Standards
```yaml
Data at Rest:
  Database: Transparent Data Encryption (TDE)
  File System: LUKS encryption
  Backups: AES-256-GCM
  Keys: Stored separately

Data in Transit:
  External: TLS 1.2+ minimum
  Internal: TLS for database connections
  API: HTTPS only
  Monitoring: Encrypted metrics

Sensitive Data Handling:
  Passwords: Bcrypt (cost factor 12)
  API Keys: Encrypted in database
  PII: Tokenization where possible
  Logs: Sanitized before storage
```

#### Backup Security
```yaml
Backup Strategy:
  Frequency: Daily automated
  Retention: 30 days
  Storage: Encrypted external drives
  Testing: Weekly restore test
  
Encryption:
  Algorithm: AES-256-GCM
  Key Management: Separate from backups
  Verification: SHA-256 checksums
  Access: Restricted to ops team
```

---

## Document Metadata

**Version**: 2.0  
**Last Updated**: January 2025  
**Owner**: CTO/Technical Director  
**Review Cycle**: Monthly  
**Distribution**: Technical Teams  

**Key Updates**:
- Consolidated infrastructure specifications
- Clarified MVP local deployment only
- Added detailed resource allocation
- Included future migration path
- Unified security architecture