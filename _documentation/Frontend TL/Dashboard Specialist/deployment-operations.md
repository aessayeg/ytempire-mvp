# 8. DEPLOYMENT & OPERATIONS

## Overview

### System Architecture
YTEMPIRE is an automated YouTube content generation platform supporting 100+ channels with 95% automation. The system consists of 17 team members across 5 specialized teams building a platform that generates videos at <$0.50 per video.

### Technology Stack
- **Frontend**: React 18, TypeScript, Material-UI (~300KB), Zustand (NOT Redux), Recharts (NOT D3.js)
- **Backend**: FastAPI, PostgreSQL 15, Redis 7, Celery
- **AI/ML**: GPT-4, ElevenLabs, Stable Diffusion, Custom models
- **Infrastructure**: Docker, Ubuntu 22.04, NVIDIA RTX 5090
- **Monitoring**: Prometheus, Grafana, Docker logs

### Team Structure (17 people total)
```
CEO/Founder
├── Product Owner
├── CTO/Technical Director
│   ├── Backend Team Lead (4 team)
│   │   ├── API Developer Engineer
│   │   ├── Data Pipeline Engineer
│   │   └── Integration Specialist
│   ├── Frontend Team Lead (4 team)
│   │   ├── React Engineer
│   │   ├── Dashboard Specialist
│   │   └── UI/UX Designer
│   └── Platform Ops Lead (4 team)
│       ├── DevOps Engineer
│       ├── Security Engineer
│       └── QA Engineer
└── VP of AI
    ├── AI/ML Team Lead (2 team)
    │   └── ML Engineer
    └── Data Team Lead (3 team)
        ├── Data Engineer
        └── Analytics Engineer
```

## 8.1 CI/CD Pipeline

### Overview
YTEMPIRE utilizes a Docker-based CI/CD pipeline with GitHub Actions for automated testing, building, and deployment. The pipeline supports blue-green deployments with <10 minute end-to-end deployment time and <5 minute rollback capability.

### Pipeline Architecture

```yaml
CI/CD Flow:
├── Source Control (GitHub)
│   ├── Main Branch (production)
│   ├── Develop Branch (staging)
│   └── Feature Branches (development)
│
├── GitHub Actions Pipeline
│   ├── Trigger Events
│   │   ├── Push to main/develop
│   │   ├── Pull request creation
│   │   └── Manual dispatch
│   │
│   ├── Build Stage
│   │   ├── Code checkout
│   │   ├── Dependency installation
│   │   ├── TypeScript compilation
│   │   └── Bundle optimization
│   │
│   ├── Test Stage
│   │   ├── Unit tests (Jest/Vitest)
│   │   ├── Integration tests
│   │   ├── E2E tests (Selenium)
│   │   └── Performance tests
│   │
│   ├── Quality Gates
│   │   ├── Code coverage (≥70%)
│   │   ├── Bundle size (<1MB)
│   │   ├── Security scanning
│   │   └── Linting/formatting
│   │
│   └── Deploy Stage
│       ├── Docker image build
│       ├── Registry push
│       ├── Blue-green deployment
│       └── Health check validation
```

### GitHub Actions Configuration

```yaml
# .github/workflows/deploy.yml
name: CI/CD Pipeline

on:
  push:
    branches: [main, develop]
  pull_request:
    branches: [main]

env:
  NODE_VERSION: '18.x'
  DOCKER_REGISTRY: 'local-registry:5000'

jobs:
  build:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v3
      
      - name: Setup Node.js
        uses: actions/setup-node@v3
        with:
          node-version: ${{ env.NODE_VERSION }}
          cache: 'npm'
      
      - name: Install dependencies
        run: npm ci
      
      - name: Run TypeScript compiler
        run: npm run type-check
      
      - name: Build application
        run: npm run build
        env:
          VITE_ENV: production
      
      - name: Check bundle size
        run: |
          size=$(du -sb dist | cut -f1)
          if [ $size -gt 1048576 ]; then
            echo "Bundle size exceeds 1MB limit"
            exit 1
          fi
      
      - name: Upload artifacts
        uses: actions/upload-artifact@v3
        with:
          name: dist
          path: dist/

  test:
    runs-on: ubuntu-latest
    needs: build
    steps:
      - name: Unit tests
        run: npm run test:ci
      
      - name: Integration tests
        run: npm run test:integration
      
      - name: Code coverage check
        run: |
          coverage=$(npm run test:coverage --silent | grep "All files" | awk '{print $10}' | sed 's/%//')
          if (( $(echo "$coverage < 70" | bc -l) )); then
            echo "Code coverage below 70%"
            exit 1
          fi
      
      - name: Security scan
        run: npm audit --audit-level=high

  deploy:
    runs-on: ubuntu-latest
    needs: [build, test]
    if: github.ref == 'refs/heads/main'
    steps:
      - name: Build Docker image
        run: |
          docker build -t ytempire-frontend:${{ github.sha }} .
          docker tag ytempire-frontend:${{ github.sha }} ytempire-frontend:latest
      
      - name: Push to registry
        run: |
          docker push ${{ env.DOCKER_REGISTRY }}/ytempire-frontend:${{ github.sha }}
          docker push ${{ env.DOCKER_REGISTRY }}/ytempire-frontend:latest
      
      - name: Deploy to production
        run: |
          ssh deploy@production-server 'cd /opt/ytempire && ./deploy.sh ${{ github.sha }}'
      
      - name: Health check
        run: |
          for i in {1..30}; do
            if curl -f http://production-server/health; then
              echo "Deployment successful"
              exit 0
            fi
            sleep 10
          done
          echo "Health check failed"
          exit 1
```

### Deployment Scripts

```bash
#!/bin/bash
# deploy.sh - Blue-green deployment script

set -e

VERSION=${1:-latest}
BLUE_PORT=3000
GREEN_PORT=3001
CURRENT_COLOR=$(cat /opt/ytempire/current_color.txt || echo "blue")
NEW_COLOR=$([[ "$CURRENT_COLOR" == "blue" ]] && echo "green" || echo "blue")
NEW_PORT=$([[ "$NEW_COLOR" == "blue" ]] && echo "$BLUE_PORT" || echo "$GREEN_PORT")

echo "Deploying version $VERSION to $NEW_COLOR environment..."

# Pull new image
docker pull local-registry:5000/ytempire-frontend:$VERSION

# Start new container
docker run -d \
  --name ytempire-$NEW_COLOR \
  --network ytempire-network \
  -p $NEW_PORT:3000 \
  -e NODE_ENV=production \
  -v /opt/ytempire/config:/app/config:ro \
  --restart unless-stopped \
  local-registry:5000/ytempire-frontend:$VERSION

# Wait for health check
echo "Waiting for health check..."
for i in {1..30}; do
  if curl -f http://localhost:$NEW_PORT/health; then
    echo "Health check passed"
    break
  fi
  sleep 2
done

# Switch traffic
echo "Switching traffic to $NEW_COLOR..."
sed -i "s/proxy_pass http:\/\/localhost:[0-9]*/proxy_pass http:\/\/localhost:$NEW_PORT/" /etc/nginx/sites-available/ytempire
nginx -s reload

# Update current color
echo "$NEW_COLOR" > /opt/ytempire/current_color.txt

# Stop old container
echo "Stopping $CURRENT_COLOR container..."
sleep 5
docker stop ytempire-$CURRENT_COLOR || true
docker rm ytempire-$CURRENT_COLOR || true

echo "Deployment complete!"
```

## 8.2 Deployment Process

### Local Server Infrastructure

```yaml
Hardware Specifications:
├── Primary Server
│   ├── CPU: AMD Ryzen 9 9950X3D (16 cores, 32 threads)
│   ├── RAM: 128GB DDR5-5600
│   ├── GPU: NVIDIA RTX 5090 (32GB VRAM)
│   ├── Storage:
│   │   ├── System: 2TB NVMe Gen5 (OS + Applications)
│   │   ├── Data: 4TB NVMe Gen4 (Database + Cache)
│   │   └── Media: 8TB SSD (Video storage)
│   └── Network: 1Gbps Fiber (symmetric)
│
└── Backup Infrastructure
    ├── Local: 8TB External SSD (daily backups)
    ├── Cloud: Google Drive (weekly sync)
    └── Snapshots: ZFS hourly snapshots
```

### Docker Compose Stack

```yaml
# docker-compose.yml
version: '3.8'

services:
  frontend:
    image: ytempire-frontend:latest
    container_name: ytempire-frontend
    ports:
      - "3000:3000"
    environment:
      - NODE_ENV=production
      - VITE_API_BASE_URL=http://backend:8000/api/v1
      - VITE_WS_URL=ws://backend:8001/ws
    volumes:
      - ./config:/app/config:ro
      - frontend-assets:/app/dist
    depends_on:
      - backend
    restart: unless-stopped
    healthcheck:
      test: ["CMD", "curl", "-f", "http://localhost:3000/health"]
      interval: 30s
      timeout: 10s
      retries: 3
    deploy:
      resources:
        limits:
          cpus: '2'
          memory: 2G
        reservations:
          cpus: '1'
          memory: 1G

  backend:
    image: ytempire-backend:latest
    container_name: ytempire-backend
    ports:
      - "8000:8000"
      - "8001:8001"  # WebSocket
    environment:
      - ENVIRONMENT=production
      - DATABASE_URL=postgresql://ytempire:${DB_PASSWORD}@postgres:5432/ytempire
      - REDIS_URL=redis://redis:6379
      - SECRET_KEY=${SECRET_KEY}
    volumes:
      - ./config:/app/config:ro
      - media-storage:/app/media
    depends_on:
      - postgres
      - redis
    restart: unless-stopped
    deploy:
      resources:
        limits:
          cpus: '4'
          memory: 4G

  postgres:
    image: postgres:15-alpine
    container_name: ytempire-postgres
    environment:
      - POSTGRES_USER=ytempire
      - POSTGRES_PASSWORD=${DB_PASSWORD}
      - POSTGRES_DB=ytempire
    volumes:
      - postgres-data:/var/lib/postgresql/data
      - ./backups/postgres:/backups
    ports:
      - "5432:5432"
    restart: unless-stopped
    deploy:
      resources:
        limits:
          cpus: '4'
          memory: 16G

  redis:
    image: redis:7-alpine
    container_name: ytempire-redis
    command: redis-server --appendonly yes
    volumes:
      - redis-data:/data
    ports:
      - "6379:6379"
    restart: unless-stopped
    deploy:
      resources:
        limits:
          cpus: '2'
          memory: 8G

  nginx:
    image: nginx:alpine
    container_name: ytempire-nginx
    ports:
      - "80:80"
      - "443:443"
    volumes:
      - ./nginx/nginx.conf:/etc/nginx/nginx.conf:ro
      - ./nginx/sites:/etc/nginx/sites-available:ro
      - ./ssl:/etc/nginx/ssl:ro
      - frontend-assets:/usr/share/nginx/html:ro
    depends_on:
      - frontend
      - backend
    restart: unless-stopped

  prometheus:
    image: prom/prometheus:latest
    container_name: ytempire-prometheus
    command:
      - '--config.file=/etc/prometheus/prometheus.yml'
      - '--storage.tsdb.path=/prometheus'
      - '--storage.tsdb.retention.time=7d'
    volumes:
      - ./monitoring/prometheus.yml:/etc/prometheus/prometheus.yml:ro
      - prometheus-data:/prometheus
    ports:
      - "9090:9090"
    restart: unless-stopped

  grafana:
    image: grafana/grafana:latest
    container_name: ytempire-grafana
    environment:
      - GF_SECURITY_ADMIN_USER=admin
      - GF_SECURITY_ADMIN_PASSWORD=${GRAFANA_PASSWORD}
    volumes:
      - ./monitoring/grafana/dashboards:/etc/grafana/provisioning/dashboards:ro
      - ./monitoring/grafana/datasources:/etc/grafana/provisioning/datasources:ro
      - grafana-data:/var/lib/grafana
    ports:
      - "3001:3000"
    depends_on:
      - prometheus
    restart: unless-stopped

volumes:
  frontend-assets:
  postgres-data:
  redis-data:
  media-storage:
  prometheus-data:
  grafana-data:

networks:
  default:
    name: ytempire-network
    driver: bridge
```

### Deployment Checklist

```markdown
## Pre-Deployment Checklist

### Code Preparation
- [ ] All tests passing (≥70% coverage)
- [ ] TypeScript compilation successful
- [ ] Bundle size <1MB verified
- [ ] No console.log statements in production code
- [ ] Environment variables configured
- [ ] API endpoints updated and tested
- [ ] Database migrations prepared

### Security Checks
- [ ] Security scan completed (npm audit)
- [ ] No exposed secrets or API keys
- [ ] HTTPS certificates valid
- [ ] CORS configuration correct
- [ ] Authentication flow tested
- [ ] Rate limiting configured

### Performance Validation
- [ ] Lighthouse score >85
- [ ] Page load time <2 seconds
- [ ] Time to Interactive <3 seconds
- [ ] Memory usage <200MB
- [ ] API response times <1 second

### Infrastructure Readiness
- [ ] Docker images built and tagged
- [ ] Database backup completed
- [ ] Disk space available (>20GB)
- [ ] Network connectivity verified
- [ ] SSL certificates renewed
- [ ] Monitoring alerts configured

## Deployment Steps

1. **Backup Current State**
   ```bash
   ./scripts/backup.sh full
   ```

2. **Pull Latest Code**
   ```bash
   git pull origin main
   git verify-commit HEAD
   ```

3. **Build and Test**
   ```bash
   npm ci
   npm run build
   npm run test:ci
   ```

4. **Deploy with Blue-Green**
   ```bash
   ./deploy.sh $(git rev-parse HEAD)
   ```

5. **Verify Deployment**
   ```bash
   ./scripts/health-check.sh
   curl -I https://ytempire.com
   ```

6. **Monitor for Issues**
   - Check Grafana dashboards
   - Monitor error rates
   - Watch for performance degradation
   - Review user feedback channels

## Rollback Procedure

If issues detected within 5 minutes:
```bash
./scripts/rollback.sh immediate
```

For non-critical issues:
```bash
./scripts/rollback.sh scheduled
```
```

## 8.3 Monitoring & Observability

### Monitoring Stack Architecture

```yaml
Monitoring Infrastructure:
├── Metrics Collection
│   ├── Prometheus (metrics aggregation)
│   ├── Node Exporter (system metrics)
│   ├── Custom Exporters (application metrics)
│   └── Push Gateway (batch job metrics)
│
├── Visualization
│   ├── Grafana Dashboards
│   │   ├── System Overview
│   │   ├── Application Performance
│   │   ├── Business Metrics
│   │   └── Custom Alerts
│   └── Real-time Displays
│
├── Log Management
│   ├── Docker Logs (container output)
│   ├── Application Logs (structured JSON)
│   ├── Access Logs (nginx)
│   └── Audit Logs (security events)
│
└── Alerting
    ├── Alertmanager (Prometheus)
    ├── Email Notifications
    ├── Slack Integration
    └── PagerDuty (critical only)
```

### Prometheus Configuration

```yaml
# prometheus.yml
global:
  scrape_interval: 30s
  evaluation_interval: 30s
  external_labels:
    monitor: 'ytempire-monitor'
    environment: 'production'

alerting:
  alertmanagers:
    - static_configs:
        - targets: ['alertmanager:9093']

rule_files:
  - 'alerts/*.yml'

scrape_configs:
  - job_name: 'prometheus'
    static_configs:
      - targets: ['localhost:9090']

  - job_name: 'node'
    static_configs:
      - targets: ['node-exporter:9100']

  - job_name: 'frontend'
    static_configs:
      - targets: ['frontend:3000/metrics']
    metrics_path: '/metrics'

  - job_name: 'backend'
    static_configs:
      - targets: ['backend:8000/metrics']
    metrics_path: '/api/metrics'

  - job_name: 'postgres'
    static_configs:
      - targets: ['postgres-exporter:9187']

  - job_name: 'redis'
    static_configs:
      - targets: ['redis-exporter:9121']

  - job_name: 'nginx'
    static_configs:
      - targets: ['nginx-exporter:9113']
```

### Alert Rules

```yaml
# alerts/application.yml
groups:
  - name: application
    interval: 30s
    rules:
      - alert: HighErrorRate
        expr: rate(http_requests_total{status=~"5.."}[5m]) > 0.05
        for: 5m
        labels:
          severity: critical
        annotations:
          summary: "High error rate detected"
          description: "Error rate is {{ $value }} errors per second"

      - alert: SlowAPIResponse
        expr: histogram_quantile(0.95, http_request_duration_seconds_bucket) > 1
        for: 10m
        labels:
          severity: warning
        annotations:
          summary: "API response time degraded"
          description: "95th percentile response time is {{ $value }} seconds"

      - alert: HighMemoryUsage
        expr: (node_memory_MemTotal_bytes - node_memory_MemAvailable_bytes) / node_memory_MemTotal_bytes > 0.9
        for: 5m
        labels:
          severity: warning
        annotations:
          summary: "High memory usage"
          description: "Memory usage is above 90%"

      - alert: DiskSpaceLow
        expr: node_filesystem_avail_bytes{mountpoint="/"} / node_filesystem_size_bytes{mountpoint="/"} < 0.1
        for: 5m
        labels:
          severity: critical
        annotations:
          summary: "Low disk space"
          description: "Less than 10% disk space remaining"

      - alert: CostPerVideoHigh
        expr: avg_over_time(cost_per_video[1h]) > 0.50
        for: 30m
        labels:
          severity: warning
        annotations:
          summary: "Cost per video exceeds threshold"
          description: "Average cost per video is ${{ $value }}"

      - alert: VideoGenerationFailed
        expr: increase(video_generation_failures_total[1h]) > 5
        for: 5m
        labels:
          severity: critical
        annotations:
          summary: "Multiple video generation failures"
          description: "{{ $value }} videos failed in the last hour"
```

### Grafana Dashboard Configuration

```json
{
  "dashboard": {
    "title": "YTEMPIRE System Overview",
    "panels": [
      {
        "title": "Request Rate",
        "type": "graph",
        "targets": [
          {
            "expr": "sum(rate(http_requests_total[5m])) by (service)",
            "legendFormat": "{{ service }}"
          }
        ]
      },
      {
        "title": "Error Rate",
        "type": "graph",
        "targets": [
          {
            "expr": "sum(rate(http_requests_total{status=~\"5..\"}[5m])) by (service)",
            "legendFormat": "{{ service }}"
          }
        ]
      },
      {
        "title": "Response Time (p95)",
        "type": "graph",
        "targets": [
          {
            "expr": "histogram_quantile(0.95, sum(rate(http_request_duration_seconds_bucket[5m])) by (le, service))",
            "legendFormat": "{{ service }}"
          }
        ]
      },
      {
        "title": "Active Channels",
        "type": "stat",
        "targets": [
          {
            "expr": "active_channels_total"
          }
        ]
      },
      {
        "title": "Videos Generated Today",
        "type": "stat",
        "targets": [
          {
            "expr": "increase(videos_generated_total[1d])"
          }
        ]
      },
      {
        "title": "Cost per Video",
        "type": "gauge",
        "targets": [
          {
            "expr": "avg_over_time(cost_per_video[1h])"
          }
        ],
        "thresholds": [
          {"value": 0.40, "color": "green"},
          {"value": 0.45, "color": "yellow"},
          {"value": 0.50, "color": "red"}
        ]
      }
    ]
  }
}
```

### Application Metrics Implementation

```typescript
// metrics.ts - Application metrics collection
import { Counter, Histogram, Gauge, register } from 'prom-client';

// Request metrics
export const httpRequestsTotal = new Counter({
  name: 'http_requests_total',
  help: 'Total number of HTTP requests',
  labelNames: ['method', 'route', 'status'],
  registers: [register]
});

export const httpRequestDuration = new Histogram({
  name: 'http_request_duration_seconds',
  help: 'Duration of HTTP requests in seconds',
  labelNames: ['method', 'route'],
  buckets: [0.1, 0.3, 0.5, 0.7, 1, 3, 5, 7, 10],
  registers: [register]
});

// Business metrics
export const videosGenerated = new Counter({
  name: 'videos_generated_total',
  help: 'Total number of videos generated',
  labelNames: ['channel', 'status'],
  registers: [register]
});

export const costPerVideo = new Gauge({
  name: 'cost_per_video',
  help: 'Current cost per video in dollars',
  registers: [register]
});

export const activeChannels = new Gauge({
  name: 'active_channels_total',
  help: 'Number of active channels',
  registers: [register]
});

// Middleware for automatic metric collection
export const metricsMiddleware = (req: Request, res: Response, next: NextFunction) => {
  const start = Date.now();
  
  res.on('finish', () => {
    const duration = (Date.now() - start) / 1000;
    
    httpRequestsTotal.labels(
      req.method,
      req.route?.path || req.path,
      res.statusCode.toString()
    ).inc();
    
    httpRequestDuration.labels(
      req.method,
      req.route?.path || req.path
    ).observe(duration);
  });
  
  next();
};
```

## 8.4 Disaster Recovery

### Disaster Recovery Plan

```yaml
Recovery Strategy:
├── Recovery Objectives
│   ├── RTO (Recovery Time Objective): 4 hours
│   ├── RPO (Recovery Point Objective): 1 hour
│   └── Service Level: 95% functionality
│
├── Backup Strategy
│   ├── Database
│   │   ├── Full backup: Daily at 2 AM
│   │   ├── Incremental: Every 4 hours
│   │   └── Transaction logs: Continuous
│   │
│   ├── Application Data
│   │   ├── Config files: Git versioned
│   │   ├── Media files: Hourly sync
│   │   └── User uploads: Real-time backup
│   │
│   └── System State
│       ├── Docker volumes: Daily snapshot
│       ├── System config: Weekly backup
│       └── SSL certificates: Monthly backup
│
├── Recovery Procedures
│   ├── Level 1: Service Restart (5 minutes)
│   ├── Level 2: Container Recovery (30 minutes)
│   ├── Level 3: Database Restore (2 hours)
│   └── Level 4: Full System Recovery (4 hours)
│
└── Testing Schedule
    ├── Monthly: Service restart drill
    ├── Quarterly: Database restore test
    └── Annually: Full disaster simulation
```

### Backup Scripts

```bash
#!/bin/bash
# backup.sh - Comprehensive backup script

set -e

BACKUP_DIR="/opt/backups"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
RETENTION_DAYS=30

# Function to log messages
log() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] $1" | tee -a "$BACKUP_DIR/backup.log"
}

# Database backup
backup_database() {
    log "Starting database backup..."
    
    docker exec ytempire-postgres pg_dump \
        -U ytempire \
        -d ytempire \
        --no-owner \
        --no-acl \
        -f /backups/ytempire_${TIMESTAMP}.sql
    
    # Compress backup
    gzip "$BACKUP_DIR/postgres/ytempire_${TIMESTAMP}.sql"
    
    log "Database backup completed: ytempire_${TIMESTAMP}.sql.gz"
}

# Redis backup
backup_redis() {
    log "Starting Redis backup..."
    
    docker exec ytempire-redis redis-cli BGSAVE
    sleep 5
    
    docker cp ytempire-redis:/data/dump.rdb \
        "$BACKUP_DIR/redis/dump_${TIMESTAMP}.rdb"
    
    log "Redis backup completed: dump_${TIMESTAMP}.rdb"
}

# Application data backup
backup_application() {
    log "Starting application backup..."
    
    # Docker volumes
    docker run --rm \
        -v ytempire_media-storage:/source:ro \
        -v "$BACKUP_DIR/volumes:/backup" \
        alpine tar czf /backup/media_${TIMESTAMP}.tar.gz -C /source .
    
    # Configuration files
    tar czf "$BACKUP_DIR/config/config_${TIMESTAMP}.tar.gz" \
        /opt/ytempire/config \
        /opt/ytempire/.env \
        /opt/ytempire/docker-compose.yml
    
    log "Application backup completed"
}

# Upload to cloud storage
upload_to_cloud() {
    log "Uploading to cloud storage..."
    
    rclone copy \
        "$BACKUP_DIR/postgres/ytempire_${TIMESTAMP}.sql.gz" \
        gdrive:ytempire-backups/postgres/ \
        --progress
    
    rclone copy \
        "$BACKUP_DIR/redis/dump_${TIMESTAMP}.rdb" \
        gdrive:ytempire-backups/redis/ \
        --progress
    
    log "Cloud upload completed"
}

# Cleanup old backups
cleanup_old_backups() {
    log "Cleaning up old backups..."
    
    find "$BACKUP_DIR" -type f -mtime +$RETENTION_DAYS -delete
    
    log "Cleanup completed"
}

# Main execution
main() {
    log "=== Starting backup process ==="
    
    backup_database
    backup_redis
    backup_application
    upload_to_cloud
    cleanup_old_backups
    
    log "=== Backup process completed successfully ==="
}

# Run main function
main
```

### Recovery Procedures

```bash
#!/bin/bash
# recovery.sh - Disaster recovery script

set -e

RECOVERY_TYPE=${1:-"service"}
BACKUP_DATE=${2:-"latest"}

# Service restart (Level 1)
recover_service() {
    echo "Performing service restart..."
    
    docker-compose restart
    
    # Wait for health checks
    sleep 30
    
    # Verify services
    docker-compose ps
    curl -f http://localhost:3000/health || exit 1
    
    echo "Service restart completed"
}

# Container recovery (Level 2)
recover_containers() {
    echo "Performing container recovery..."
    
    # Stop all containers
    docker-compose down
    
    # Remove corrupted volumes
    docker volume prune -f
    
    # Recreate containers
    docker-compose up -d
    
    # Wait for initialization
    sleep 60
    
    # Verify all services
    ./scripts/health-check.sh all
    
    echo "Container recovery completed"
}

# Database restore (Level 3)
recover_database() {
    echo "Performing database restore..."
    
    # Find backup file
    if [ "$BACKUP_DATE" == "latest" ]; then
        BACKUP_FILE=$(ls -t /opt/backups/postgres/*.sql.gz | head -1)
    else
        BACKUP_FILE="/opt/backups/postgres/ytempire_${BACKUP_DATE}.sql.gz"
    fi
    
    if [ ! -f "$BACKUP_FILE" ]; then
        echo "Backup file not found: $BACKUP_FILE"
        exit 1
    fi
    
    echo "Using backup: $BACKUP_FILE"
    
    # Stop application containers
    docker-compose stop frontend backend
    
    # Restore database
    gunzip -c "$BACKUP_FILE" | docker exec -i ytempire-postgres \
        psql -U ytempire -d ytempire
    
    # Restart services
    docker-compose start frontend backend
    
    # Verify restoration
    docker exec ytempire-postgres \
        psql -U ytempire -d ytempire -c "SELECT COUNT(*) FROM channels;"
    
    echo "Database restore completed"
}

# Full system recovery (Level 4)
recover_full_system() {
    echo "Performing full system recovery..."
    
    # Stop everything
    docker-compose down -v
    
    # Restore configuration
    tar xzf /opt/backups/config/config_latest.tar.gz -C /
    
    # Restore database
    recover_database
    
    # Restore Redis
    docker cp /opt/backups/redis/dump_latest.rdb ytempire-redis:/data/dump.rdb
    docker-compose restart redis
    
    # Restore media files
    docker run --rm \
        -v ytempire_media-storage:/target \
        -v /opt/backups/volumes:/backup:ro \
        alpine tar xzf /backup/media_latest.tar.gz -C /target
    
    # Start all services
    docker-compose up -d
    
    # Full health check
    sleep 120
    ./scripts/health-check.sh full
    
    echo "Full system recovery completed"
}

# Main execution
case "$RECOVERY_TYPE" in
    service)
        recover_service
        ;;
    container)
        recover_containers
        ;;
    database)
        recover_database
        ;;
    full)
        recover_full_system
        ;;
    *)
        echo "Usage: $0 {service|container|database|full} [backup_date]"
        exit 1
        ;;
esac
```

### Incident Response Procedures

```markdown
## Incident Response Runbook

### Severity Levels

| Level | Description | Response Time | Escalation |
|-------|-------------|---------------|------------|
| SEV-1 | Complete outage | Immediate | All hands |
| SEV-2 | Major degradation | 15 minutes | Team lead + on-call |
| SEV-3 | Minor issue | 1 hour | On-call engineer |
| SEV-4 | Non-critical | Next business day | Regular queue |

### Response Procedures

#### SEV-1: Complete Outage

1. **Immediate Actions (0-5 minutes)**
   - Acknowledge incident in PagerDuty
   - Join incident channel in Slack
   - Run initial diagnostics:
     ```bash
     ./scripts/diagnose.sh full
     ```

2. **Assessment (5-15 minutes)**
   - Identify root cause
   - Determine impact scope
   - Estimate recovery time
   - Update status page

3. **Recovery (15+ minutes)**
   - Execute appropriate recovery procedure
   - Monitor recovery progress
   - Verify service restoration
   - Update stakeholders

4. **Post-Incident (within 24 hours)**
   - Document timeline
   - Identify root cause
   - Create action items
   - Schedule postmortem

#### SEV-2: Major Degradation

1. **Initial Response (0-15 minutes)**
   - Check monitoring dashboards
   - Identify affected services
   - Assess user impact

2. **Mitigation (15-60 minutes)**
   - Apply temporary fixes
   - Scale resources if needed
   - Enable degraded mode
   - Communicate with users

3. **Resolution**
   - Implement permanent fix
   - Verify full functionality
   - Update documentation

### Communication Templates

#### Status Page Update
```
Title: [Service Name] Experiencing Issues
Status: Investigating | Identified | Monitoring | Resolved
 
We are currently experiencing [describe issue].
Impact: [describe user impact]
Next Update: [time]
 
Timeline:
[HH:MM] - Issue detected
[HH:MM] - Root cause identified
[HH:MM] - Fix implemented
[HH:MM] - Service restored
```

#### Internal Escalation
```
INCIDENT: [SEV-X] [Brief Description]
Time Detected: [timestamp]
Services Affected: [list]
User Impact: [description]
Current Status: [investigating/mitigating/resolved]
Incident Commander: [name]
Slack Channel: #incident-[timestamp]
```
```

## 8.5 Performance Monitoring

### Dashboard Specialist Requirements

The Dashboard Specialist is the sole owner of all data visualization and real-time monitoring interfaces. This role requires unique expertise in handling complex data at scale:

#### Core Responsibilities
- **Data Visualization Mastery**: Transform 1M+ data points into actionable insights
- **Real-Time Complexity**: Handle 100+ YouTube channels generating concurrent data streams
- **Performance Engineering**: Achieve 60fps interactions with millions of data points
- **Innovation Opportunity**: Create industry-leading dashboard visualizations

#### Technical Requirements
```yaml
Dashboard Performance Targets:
  Data Scale:
    - Concurrent channels: 100+ displayed simultaneously
    - Data points per chart: Up to 1M (with virtualization)
    - Update frequency: Sub-second for critical metrics
    - Data streams: 20+ real-time feeds
  
  Performance Metrics:
    - Initial render: <500ms
    - Data refresh: <1 second
    - Interaction latency: <100ms
    - Memory usage: <200MB
    - Frame rate: 60fps maintained
  
  Visualization Requirements:
    - Chart types: 5-7 Recharts components
    - Custom visualizations: Permitted within Recharts
    - Drill-down depth: 3 levels minimum
    - Export formats: CSV, JSON, PNG
    - Accessibility: WCAG AA compliant
```

#### Dashboard Architecture Ownership
```typescript
// Dashboard Specialist Domain
const dashboardOwnership = {
  analytics: {
    channelPerformance: ['real-time metrics', 'historical trends'],
    revenueAnalytics: ['earnings', 'forecasting', 'ROI tracking'],
    contentAnalytics: ['video performance', 'viral tracking'],
    audienceInsights: ['demographics', 'retention', 'growth']
  },
  
  operational: {
    systemMonitoring: ['platform health', 'API performance'],
    workflowManagement: ['content pipeline', 'automation health'],
    multiChannelOverview: ['bulk operations', 'alert center']
  },
  
  technical: {
    dataAggregation: 'Client-side for performance',
    caching: 'Strategic use of memoization',
    virtualization: 'For lists >100 items',
    webWorkers: 'For heavy calculations'
  }
};
```

### Performance Monitoring Implementation

```yaml
Performance Monitoring:
├── Frontend Metrics
│   ├── Core Web Vitals
│   │   ├── LCP (Largest Contentful Paint): <2.5s
│   │   ├── FID (First Input Delay): <100ms
│   │   ├── CLS (Cumulative Layout Shift): <0.1
│   │   └── FCP (First Contentful Paint): <1.8s
│   │
│   ├── Custom Metrics
│   │   ├── Time to Interactive: <3s
│   │   ├── Dashboard Load Time: <2s
│   │   ├── API Response Time: <1s
│   │   └── Bundle Size: <1MB
│   │
│   └── User Experience
│       ├── Page Load Speed
│       ├── Interaction Latency
│       ├── Error Rate
│       └── Session Duration
│
├── Backend Metrics
│   ├── API Performance
│   │   ├── Request Latency
│   │   ├── Throughput (RPS)
│   │   ├── Error Rate
│   │   └── Success Rate
│   │
│   ├── Database Performance
│   │   ├── Query Time
│   │   ├── Connection Pool
│   │   ├── Lock Contention
│   │   └── Cache Hit Rate
│   │
│   └── Resource Utilization
│       ├── CPU Usage
│       ├── Memory Usage
│       ├── Disk I/O
│       └── Network I/O
│
└── Business Metrics
    ├── Video Generation
    │   ├── Generation Time
    │   ├── Success Rate
    │   ├── Cost per Video
    │   └── Quality Score
    │
    └── Platform Health
        ├── Active Users
        ├── Channel Activity
        ├── Revenue Metrics
        └── Error Budgets
```

### Performance Monitoring Implementation

```typescript
// performance-monitor.ts
import { getCLS, getFID, getFCP, getLCP, getTTFB } from 'web-vitals';

class PerformanceMonitor {
  private metrics: Map<string, number[]> = new Map();
  private reportingEndpoint = '/api/metrics/performance';
  
  initialize() {
    // Core Web Vitals
    getCLS(this.handleMetric.bind(this));
    getFID(this.handleMetric.bind(this));
    getFCP(this.handleMetric.bind(this));
    getLCP(this.handleMetric.bind(this));
    getTTFB(this.handleMetric.bind(this));
    
    // Custom metrics
    this.measureDashboardLoad();
    this.measureAPILatency();
    this.trackInteractions();
    
    // Report metrics every 30 seconds
    setInterval(() => this.reportMetrics(), 30000);
  }
  
  private handleMetric(metric: Metric) {
    const values = this.metrics.get(metric.name) || [];
    values.push(metric.value);
    this.metrics.set(metric.name, values);
    
    // Alert on threshold violations
    this.checkThresholds(metric);
  }
  
  private checkThresholds(metric: Metric) {
    const thresholds = {
      LCP: 2500,
      FID: 100,
      CLS: 0.1,
      FCP: 1800,
      TTFB: 600
    };
    
    if (metric.value > thresholds[metric.name]) {
      console.warn(`Performance degradation: ${metric.name} = ${metric.value}ms`);
      this.sendAlert(metric);
    }
  }
  
  private measureDashboardLoad() {
    const observer = new PerformanceObserver((list) => {
      for (const entry of list.getEntries()) {
        if (entry.name.includes('dashboard')) {
          this.handleMetric({
            name: 'dashboard_load',
            value: entry.duration,
            id: entry.entryType,
            entries: []
          });
        }
      }
    });
    
    observer.observe({ entryTypes: ['navigation', 'resource'] });
  }
  
  private measureAPILatency() {
    const originalFetch = window.fetch;
    
    window.fetch = async (...args) => {
      const startTime = performance.now();
      
      try {
        const response = await originalFetch(...args);
        const duration = performance.now() - startTime;
        
        this.handleMetric({
          name: 'api_latency',
          value: duration,
          id: args[0].toString(),
          entries: []
        });
        
        return response;
      } catch (error) {
        const duration = performance.now() - startTime;
        
        this.handleMetric({
          name: 'api_error',
          value: duration,
          id: args[0].toString(),
          entries: []
        });
        
        throw error;
      }
    };
  }
  
  private trackInteractions() {
    let interactionCount = 0;
    
    ['click', 'keydown', 'scroll'].forEach(eventType => {
      document.addEventListener(eventType, () => {
        interactionCount++;
      }, { passive: true });
    });
    
    setInterval(() => {
      this.handleMetric({
        name: 'user_interactions',
        value: interactionCount,
        id: 'interaction_rate',
        entries: []
      });
      interactionCount = 0;
    }, 60000);
  }
  
  private async reportMetrics() {
    const report = {};
    
    this.metrics.forEach((values, name) => {
      report[name] = {
        avg: values.reduce((a, b) => a + b, 0) / values.length,
        min: Math.min(...values),
        max: Math.max(...values),
        p95: this.calculatePercentile(values, 95),
        count: values.length
      };
    });
    
    try {
      await fetch(this.reportingEndpoint, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(report)
      });
      
      // Clear metrics after reporting
      this.metrics.clear();
    } catch (error) {
      console.error('Failed to report metrics:', error);
    }
  }
  
  private calculatePercentile(values: number[], percentile: number): number {
    const sorted = [...values].sort((a, b) => a - b);
    const index = Math.ceil((percentile / 100) * sorted.length) - 1;
    return sorted[index];
  }
  
  private sendAlert(metric: Metric) {
    // Send to monitoring system
    fetch('/api/alerts/performance', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        metric: metric.name,
        value: metric.value,
        threshold: this.getThreshold(metric.name),
        timestamp: Date.now()
      })
    });
  }
  
  private getThreshold(metricName: string): number {
    const thresholds = {
      LCP: 2500,
      FID: 100,
      CLS: 0.1,
      FCP: 1800,
      TTFB: 600,
      dashboard_load: 2000,
      api_latency: 1000
    };
    return thresholds[metricName] || 0;
  }
}

// Initialize performance monitoring
export const performanceMonitor = new PerformanceMonitor();
performanceMonitor.initialize();
```

### Performance Dashboard Configuration

```yaml
# grafana-performance-dashboard.yml
dashboard:
  title: "YTEMPIRE Performance Metrics"
  refresh: "30s"
  time: 
    from: "now-6h"
    to: "now"
  
  panels:
    - title: "Core Web Vitals"
      type: "graph"
      gridPos: { h: 8, w: 12, x: 0, y: 0 }
      targets:
        - expr: "histogram_quantile(0.75, web_vitals_lcp_bucket)"
          legendFormat: "LCP (p75)"
        - expr: "histogram_quantile(0.75, web_vitals_fid_bucket)"
          legendFormat: "FID (p75)"
        - expr: "histogram_quantile(0.75, web_vitals_cls_bucket)"
          legendFormat: "CLS (p75)"
    
    - title: "API Latency Distribution"
      type: "heatmap"
      gridPos: { h: 8, w: 12, x: 12, y: 0 }
      targets:
        - expr: "sum(increase(api_latency_bucket[1m])) by (le)"
    
    - title: "Dashboard Load Times"
      type: "graph"
      gridPos: { h: 8, w: 12, x: 0, y: 8 }
      targets:
        - expr: "dashboard_load_time_seconds"
          legendFormat: "{{ page }}"
      thresholds:
        - value: 2
          color: "red"
          op: "gt"
    
    - title: "Error Rate"
      type: "stat"
      gridPos: { h: 4, w: 6, x: 12, y: 8 }
      targets:
        - expr: "sum(rate(http_requests_total{status=~'5..'}[5m]))"
      thresholds:
        - value: 0.01
          color: "yellow"
        - value: 0.05
          color: "red"
    
    - title: "Request Rate"
      type: "stat"
      gridPos: { h: 4, w: 6, x: 18, y: 8 }
      targets:
        - expr: "sum(rate(http_requests_total[5m]))"
    
    - title: "Memory Usage"
      type: "graph"
      gridPos: { h: 8, w: 12, x: 0, y: 16 }
      targets:
        - expr: "process_resident_memory_bytes / 1024 / 1024"
          legendFormat: "{{ service }}"
      alert:
        condition: "above"
        threshold: 200
    
    - title: "Video Generation Performance"
      type: "graph"
      gridPos: { h: 8, w: 12, x: 12, y: 16 }
      targets:
        - expr: "video_generation_duration_seconds"
          legendFormat: "Generation Time"
        - expr: "video_generation_cost_dollars"
          legendFormat: "Cost per Video"
```

### Performance Optimization Checklist

```markdown
## Performance Optimization Checklist

### Frontend Optimization
- [ ] Code splitting implemented
- [ ] Lazy loading for routes
- [ ] Image optimization (WebP, responsive images)
- [ ] Bundle size <1MB verified
- [ ] Tree shaking enabled
- [ ] CSS purging configured
- [ ] Service worker caching
- [ ] Preload/prefetch critical resources
- [ ] Minimize render-blocking resources
- [ ] Compress assets (gzip/brotli)

### Backend Optimization
- [ ] Database indexes optimized
- [ ] Query optimization completed
- [ ] Connection pooling configured
- [ ] Caching strategy implemented
- [ ] Rate limiting enabled
- [ ] Pagination for large datasets
- [ ] Async processing for heavy tasks
- [ ] API response compression
- [ ] CDN for static assets
- [ ] Load balancing configured

### Infrastructure Optimization
- [ ] Resource limits set
- [ ] Auto-scaling configured
- [ ] Health checks optimized
- [ ] Log rotation enabled
- [ ] Monitoring overhead minimized
- [ ] Network optimization
- [ ] Disk I/O optimization
- [ ] Memory management tuned
- [ ] CPU affinity set
- [ ] Container optimization

### Monitoring & Alerting
- [ ] Performance budgets defined
- [ ] Alert thresholds configured
- [ ] Dashboard visibility adequate
- [ ] Synthetic monitoring active
- [ ] Real user monitoring (RUM)
- [ ] Error tracking enabled
- [ ] Performance regression detection
- [ ] Capacity planning metrics
- [ ] SLA compliance tracking
- [ ] Cost optimization metrics
```

---

*End of Section 8: DEPLOYMENT & OPERATIONS*