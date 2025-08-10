# YTEMPIRE Documentation - Infrastructure & Operations

## 5.1 Infrastructure Management

### Local Server Infrastructure (MVP Phase)

#### Hardware Specifications
```yaml
Server Configuration:
  Model: Custom Build
  
  CPU:
    Model: AMD Ryzen 9 9950X3D
    Cores: 16 physical (32 threads)
    Base Clock: 4.3 GHz
    Boost Clock: 5.7 GHz
    Cache: 144MB total
  
  Memory:
    Capacity: 128GB
    Type: DDR5-6000
    Configuration: 4x32GB
    ECC: No (consumer platform)
  
  GPU:
    Model: NVIDIA RTX 5090
    VRAM: 32GB GDDR7
    CUDA Cores: 21,760
    Tensor Cores: 680
    RT Cores: 170
  
  Storage:
    System Drive: 2TB Samsung 990 Pro (NVMe Gen5)
    Data Drive: 8TB Samsung 990 Pro (NVMe Gen4)
    Backup Drive: 8TB External USB 3.2
    
  Network:
    Primary: 1Gbps Fiber (symmetric)
    Backup: 5G Mobile Hotspot
    Router: Enterprise-grade with QoS
  
  Power:
    PSU: 1200W 80+ Platinum
    UPS: 1500VA Battery Backup
    Runtime: 15 minutes at full load
  
  Cooling:
    CPU: 360mm AIO Liquid Cooler
    Case: High airflow with 6+ fans
    GPU: Stock cooling (3-fan design)
```

#### Operating System Configuration
```bash
# Ubuntu 22.04 LTS Server Configuration

# System Optimization
echo 'vm.swappiness=10' >> /etc/sysctl.conf
echo 'net.core.rmem_max=134217728' >> /etc/sysctl.conf
echo 'net.core.wmem_max=134217728' >> /etc/sysctl.conf
echo 'net.ipv4.tcp_rmem=4096 87380 134217728' >> /etc/sysctl.conf
echo 'net.ipv4.tcp_wmem=4096 65536 134217728' >> /etc/sysctl.conf

# NVIDIA Driver Installation
apt-get update
apt-get install -y nvidia-driver-545 nvidia-cuda-toolkit

# Docker Installation
curl -fsSL https://get.docker.com -o get-docker.sh
sh get-docker.sh
apt-get install -y docker-compose-plugin

# GPU Support for Docker
distribution=$(. /etc/os-release;echo $ID$VERSION_ID)
curl -s -L https://nvidia.github.io/nvidia-docker/gpgkey | apt-key add -
curl -s -L https://nvidia.github.io/nvidia-docker/$distribution/nvidia-docker.list | tee /etc/apt/sources.list.d/nvidia-docker.list
apt-get update && apt-get install -y nvidia-container-toolkit
systemctl restart docker
```

#### Resource Allocation Strategy
```yaml
CPU Allocation (16 cores):
  PostgreSQL:
    Cores: 4
    Priority: High
    Isolation: Yes
    
  Backend Services:
    Cores: 4
    Priority: Normal
    Scaling: Horizontal
    
  Video Processing:
    Cores: 4
    Priority: High
    GPU Affinity: Yes
    
  Frontend/Nginx:
    Cores: 2
    Priority: Normal
    
  Monitoring/System:
    Cores: 2
    Priority: Low

Memory Allocation (128GB):
  PostgreSQL:
    Reserved: 16GB
    Shared Buffers: 4GB
    Effective Cache: 12GB
    
  Redis:
    Reserved: 8GB
    Max Memory: 7GB
    Eviction: allkeys-lru
    
  Backend Services:
    Reserved: 24GB
    Per Container: 4GB max
    
  Video Processing:
    Reserved: 48GB
    GPU Shared: Yes
    
  Frontend:
    Reserved: 8GB
    
  System/Buffer:
    Reserved: 24GB

Storage Layout:
  /: 200GB (System)
  /var/lib/docker: 300GB (Containers)
  /data/postgres: 300GB (Database)
  /data/redis: 50GB (Cache)
  /data/videos: 6TB (Media)
  /backup: 1TB (Backups)
  /logs: 1TB (Logging)
```

### Cloud Migration Path (Post-MVP)

#### Phase 1: Hybrid Infrastructure (Months 4-5)
```yaml
Hybrid Architecture:
  Local Server:
    - GPU workloads (video processing)
    - Development/testing
    - Backup processing
    
  Cloud Services (GCP):
    - Web applications (API, Frontend)
    - Managed databases (Cloud SQL)
    - Static asset hosting (Cloud Storage)
    - CDN (Cloud CDN)
    
  Migration Strategy:
    Week 1: Cloud account setup, networking
    Week 2: Database replication to cloud
    Week 3: Web services deployment
    Week 4: Traffic migration (blue-green)
```

#### Phase 2: Full Cloud Migration (Month 6+)
```yaml
GCP Infrastructure:
  Compute:
    GKE Cluster:
      - Region: us-central1
      - Node Pools:
        - Web: n2-standard-4 (3-10 nodes)
        - GPU: n1-standard-8 + T4 (0-5 nodes)
        - Workers: n2-standard-8 (2-8 nodes)
      - Autoscaling: Enabled
      - Preemptible: 60% of nodes
    
  Storage:
    Cloud SQL:
      - Type: PostgreSQL 15
      - Tier: db-custom-4-16384
      - HA: Regional
      - Backups: Automated daily
    
    Memorystore:
      - Type: Redis 7
      - Tier: Standard
      - Size: 10GB
      - Replicas: 1
    
    Cloud Storage:
      - Videos: Standard tier
      - Archives: Nearline tier
      - Backups: Coldline tier
  
  Networking:
    - VPC: Custom mode
    - Subnets: /24 per service
    - Cloud NAT: For outbound
    - Cloud Armor: DDoS protection
    - Global Load Balancer: HTTPS

Monthly Costs (Estimated):
  - Compute: $1,200
  - Storage: $300
  - Network: $200
  - Databases: $400
  - Total: ~$2,100/month
```

### Container Management

#### Docker Compose Configuration
```yaml
# docker-compose.yml
version: '3.9'

x-common-variables: &common-variables
  TZ: UTC
  LOG_LEVEL: ${LOG_LEVEL:-info}
  ENVIRONMENT: ${ENVIRONMENT:-production}

x-healthcheck-defaults: &healthcheck-defaults
  interval: 30s
  timeout: 10s
  retries: 3
  start_period: 40s

services:
  # API Service
  api:
    image: ytempire-api:${VERSION:-latest}
    container_name: ytempire-api
    restart: unless-stopped
    ports:
      - "8080:8080"
    environment:
      <<: *common-variables
      DATABASE_URL: postgresql://ytempire:${DB_PASSWORD}@postgres:5432/ytempire
      REDIS_URL: redis://redis:6379/0
      JWT_SECRET: ${JWT_SECRET}
      OPENAI_API_KEY: ${OPENAI_API_KEY}
    volumes:
      - api-data:/app/data
      - ./logs/api:/app/logs
    depends_on:
      postgres:
        condition: service_healthy
      redis:
        condition: service_healthy
    healthcheck:
      <<: *healthcheck-defaults
      test: ["CMD", "curl", "-f", "http://localhost:8080/health"]
    deploy:
      resources:
        limits:
          cpus: '4'
          memory: 4G
        reservations:
          cpus: '2'
          memory: 2G

  # Frontend Service
  frontend:
    image: ytempire-frontend:${VERSION:-latest}
    container_name: ytempire-frontend
    restart: unless-stopped
    ports:
      - "3000:80"
    environment:
      <<: *common-variables
      API_URL: http://api:8080
    volumes:
      - ./nginx/nginx.conf:/etc/nginx/nginx.conf:ro
      - frontend-static:/usr/share/nginx/html
    healthcheck:
      <<: *healthcheck-defaults
      test: ["CMD", "curl", "-f", "http://localhost:80/health"]
    deploy:
      resources:
        limits:
          cpus: '2'
          memory: 2G

  # PostgreSQL Database
  postgres:
    image: postgres:15-alpine
    container_name: ytempire-postgres
    restart: unless-stopped
    ports:
      - "5432:5432"
    environment:
      POSTGRES_USER: ytempire
      POSTGRES_PASSWORD: ${DB_PASSWORD}
      POSTGRES_DB: ytempire
      POSTGRES_INITDB_ARGS: "--encoding=UTF8 --locale=C"
    volumes:
      - postgres-data:/var/lib/postgresql/data
      - ./init-scripts:/docker-entrypoint-initdb.d:ro
      - ./backup/postgres:/backup
    healthcheck:
      <<: *healthcheck-defaults
      test: ["CMD-SHELL", "pg_isready -U ytempire -d ytempire"]
    deploy:
      resources:
        limits:
          cpus: '4'
          memory: 16G
        reservations:
          cpus: '4'
          memory: 16G
    command: >
      postgres
      -c shared_buffers=4GB
      -c effective_cache_size=12GB
      -c maintenance_work_mem=1GB
      -c checkpoint_completion_target=0.9
      -c wal_buffers=16MB
      -c default_statistics_target=100
      -c random_page_cost=1.1
      -c effective_io_concurrency=200
      -c work_mem=10MB
      -c min_wal_size=1GB
      -c max_wal_size=4GB

  # Redis Cache
  redis:
    image: redis:7-alpine
    container_name: ytempire-redis
    restart: unless-stopped
    ports:
      - "6379:6379"
    volumes:
      - redis-data:/data
      - ./redis/redis.conf:/usr/local/etc/redis/redis.conf:ro
    healthcheck:
      <<: *healthcheck-defaults
      test: ["CMD", "redis-cli", "ping"]
    deploy:
      resources:
        limits:
          cpus: '2'
          memory: 8G
    command: redis-server /usr/local/etc/redis/redis.conf

  # Celery Worker for Video Processing
  worker:
    image: ytempire-worker:${VERSION:-latest}
    container_name: ytempire-worker
    restart: unless-stopped
    environment:
      <<: *common-variables
      DATABASE_URL: postgresql://ytempire:${DB_PASSWORD}@postgres:5432/ytempire
      REDIS_URL: redis://redis:6379/1
      OPENAI_API_KEY: ${OPENAI_API_KEY}
      ELEVENLABS_API_KEY: ${ELEVENLABS_API_KEY}
    volumes:
      - ./videos:/app/videos
      - ./logs/worker:/app/logs
    depends_on:
      - postgres
      - redis
    deploy:
      resources:
        limits:
          cpus: '4'
          memory: 48G
        reservations:
          cpus: '2'
          memory: 16G
          devices:
            - driver: nvidia
              count: 1
              capabilities: [gpu]

  # N8N Automation
  n8n:
    image: n8nio/n8n:latest
    container_name: ytempire-n8n
    restart: unless-stopped
    ports:
      - "5678:5678"
    environment:
      <<: *common-variables
      N8N_BASIC_AUTH_ACTIVE: "true"
      N8N_BASIC_AUTH_USER: ${N8N_USER}
      N8N_BASIC_AUTH_PASSWORD: ${N8N_PASSWORD}
      WEBHOOK_URL: http://n8n:5678/
    volumes:
      - n8n-data:/home/node/.n8n
      - ./n8n-workflows:/home/node/.n8n/workflows
    healthcheck:
      <<: *healthcheck-defaults
      test: ["CMD", "curl", "-f", "http://localhost:5678/healthz"]
    deploy:
      resources:
        limits:
          cpus: '2'
          memory: 8G

  # Nginx Reverse Proxy
  nginx:
    image: nginx:alpine
    container_name: ytempire-nginx
    restart: unless-stopped
    ports:
      - "80:80"
      - "443:443"
    volumes:
      - ./nginx/conf.d:/etc/nginx/conf.d:ro
      - ./nginx/ssl:/etc/nginx/ssl:ro
      - ./logs/nginx:/var/log/nginx
    depends_on:
      - api
      - frontend
    healthcheck:
      <<: *healthcheck-defaults
      test: ["CMD", "curl", "-f", "http://localhost/health"]

  # Prometheus Monitoring
  prometheus:
    image: prom/prometheus:latest
    container_name: ytempire-prometheus
    restart: unless-stopped
    ports:
      - "9090:9090"
    volumes:
      - ./prometheus/prometheus.yml:/etc/prometheus/prometheus.yml:ro
      - ./prometheus/alerts:/etc/prometheus/alerts:ro
      - prometheus-data:/prometheus
    command:
      - '--config.file=/etc/prometheus/prometheus.yml'
      - '--storage.tsdb.path=/prometheus'
      - '--storage.tsdb.retention.time=30d'
      - '--web.enable-lifecycle'
    deploy:
      resources:
        limits:
          cpus: '1'
          memory: 2G

  # Grafana Dashboard
  grafana:
    image: grafana/grafana:latest
    container_name: ytempire-grafana
    restart: unless-stopped
    ports:
      - "3001:3000"
    environment:
      GF_SECURITY_ADMIN_USER: ${GRAFANA_USER}
      GF_SECURITY_ADMIN_PASSWORD: ${GRAFANA_PASSWORD}
      GF_USERS_ALLOW_SIGN_UP: "false"
    volumes:
      - ./grafana/provisioning:/etc/grafana/provisioning:ro
      - ./grafana/dashboards:/var/lib/grafana/dashboards:ro
      - grafana-data:/var/lib/grafana
    depends_on:
      - prometheus
    deploy:
      resources:
        limits:
          cpus: '1'
          memory: 1G

volumes:
  postgres-data:
    driver: local
  redis-data:
    driver: local
  api-data:
    driver: local
  frontend-static:
    driver: local
  n8n-data:
    driver: local
  prometheus-data:
    driver: local
  grafana-data:
    driver: local

networks:
  default:
    driver: bridge
    ipam:
      config:
        - subnet: 172.20.0.0/16