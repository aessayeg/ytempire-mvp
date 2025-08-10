# 9. OPERATIONS & DEPLOYMENT - YTEMPIRE

**Version 2.0 | January 2025**  
**Document Status: Consolidated & Standardized**  
**Last Updated: January 2025**

---

## 9.1 Deployment Process

### Deployment Pipeline

```yaml
Environments:
  Development:
    - Local development machines
    - Feature branch deployments
    - Automatic on push to feature/*
  
  Staging:
    - Staging server
    - Integration testing
    - Automatic on merge to develop
  
  Production:
    - Production server
    - Manual approval required
    - Deploy from main branch only

Pipeline Stages:
  1. Code Quality Check
  2. Unit Tests
  3. Integration Tests
  4. Build Docker Images
  5. Security Scanning
  6. Deploy to Environment
  7. Health Checks
  8. Rollback if Failed
```

### CI/CD Configuration

```yaml
# .github/workflows/deploy.yml
name: Deploy Pipeline

on:
  push:
    branches: [main, develop]
  pull_request:
    branches: [main]

jobs:
  test:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v3
      
      - name: Set up Python
        uses: actions/setup-python@v4
        with:
          python-version: '3.11'
      
      - name: Install dependencies
        run: |
          pip install -r requirements.txt
          pip install -r requirements-dev.txt
      
      - name: Run linting
        run: |
          black --check app/
          flake8 app/
          mypy app/
      
      - name: Run tests
        run: |
          pytest tests/ --cov=app --cov-report=xml
      
      - name: Upload coverage
        uses: codecov/codecov-action@v3

  build:
    needs: test
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v3
      
      - name: Build Docker image
        run: |
          docker build -t ytempire/api:${{ github.sha }} .
          docker tag ytempire/api:${{ github.sha }} ytempire/api:latest
      
      - name: Security scan
        uses: aquasecurity/trivy-action@master
        with:
          image-ref: ytempire/api:${{ github.sha }}
          severity: 'CRITICAL,HIGH'
      
      - name: Push to registry
        if: github.ref == 'refs/heads/main'
        run: |
          echo ${{ secrets.DOCKER_PASSWORD }} | docker login -u ${{ secrets.DOCKER_USERNAME }} --password-stdin
          docker push ytempire/api:${{ github.sha }}
          docker push ytempire/api:latest

  deploy:
    needs: build
    if: github.ref == 'refs/heads/main'
    runs-on: ubuntu-latest
    steps:
      - name: Deploy to production
        uses: appleboy/ssh-action@v0.1.5
        with:
          host: ${{ secrets.PRODUCTION_HOST }}
          username: ${{ secrets.PRODUCTION_USER }}
          key: ${{ secrets.PRODUCTION_SSH_KEY }}
          script: |
            cd /opt/ytempire
            docker-compose pull
            docker-compose up -d --no-deps api
            docker-compose run --rm api alembic upgrade head
            ./scripts/health_check.sh
```

### Deployment Script

```bash
#!/bin/bash
# scripts/deploy.sh

set -e

echo "🚀 Starting YTEMPIRE deployment..."

# Configuration
ENVIRONMENT=${1:-staging}
VERSION=${2:-latest}

# Color codes
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Functions
log_info() {
    echo -e "${GREEN}[INFO]${NC} $1"
}

log_warn() {
    echo -e "${YELLOW}[WARN]${NC} $1"
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Pre-deployment checks
log_info "Running pre-deployment checks..."

# Check disk space
DISK_USAGE=$(df -h / | awk 'NR==2 {print int($5)}')
if [ $DISK_USAGE -gt 80 ]; then
    log_warn "Disk usage is at ${DISK_USAGE}%"
fi

# Backup database
log_info "Backing up database..."
pg_dump $DATABASE_URL > "/backups/pre-deploy-$(date +%Y%m%d-%H%M%S).sql"

# Pull latest images
log_info "Pulling Docker images..."
docker-compose pull

# Run migrations
log_info "Running database migrations..."
docker-compose run --rm api alembic upgrade head

# Deploy services
log_info "Deploying services..."
docker-compose up -d --no-deps --scale api=2 api

# Wait for health checks
log_info "Waiting for services to be healthy..."
sleep 10

# Health check
HEALTH_CHECK=$(curl -s http://localhost:8000/health | jq -r '.status')
if [ "$HEALTH_CHECK" != "healthy" ]; then
    log_error "Health check failed!"
    log_info "Rolling back..."
    docker-compose up -d --no-deps --scale api=1 api
    exit 1
fi

# Cleanup old images
log_info "Cleaning up old images..."
docker image prune -f

log_info "✅ Deployment completed successfully!"
```

---

## 9.2 Monitoring & Logging

### Monitoring Stack

```yaml
# docker-compose.monitoring.yml
version: '3.8'

services:
  prometheus:
    image: prom/prometheus:latest
    ports:
      - "9090:9090"
    volumes:
      - ./prometheus.yml:/etc/prometheus/prometheus.yml
      - prometheus_data:/prometheus
    command:
      - '--config.file=/etc/prometheus/prometheus.yml'
      - '--storage.tsdb.path=/prometheus'
    
  grafana:
    image: grafana/grafana:latest
    ports:
      - "3001:3000"
    volumes:
      - grafana_data:/var/lib/grafana
      - ./grafana/dashboards:/etc/grafana/provisioning/dashboards
      - ./grafana/datasources:/etc/grafana/provisioning/datasources
    environment:
      - GF_SECURITY_ADMIN_PASSWORD=admin
      - GF_INSTALL_PLUGINS=redis-datasource
  
  loki:
    image: grafana/loki:latest
    ports:
      - "3100:3100"
    volumes:
      - ./loki-config.yaml:/etc/loki/local-config.yaml
      - loki_data:/loki
  
  promtail:
    image: grafana/promtail:latest
    volumes:
      - /var/log:/var/log:ro
      - ./promtail-config.yaml:/etc/promtail/config.yml
      - /var/lib/docker/containers:/var/lib/docker/containers:ro

volumes:
  prometheus_data:
  grafana_data:
  loki_data:
```

### Application Metrics

```python
# app/core/metrics.py
from prometheus_client import Counter, Histogram, Gauge, generate_latest, CONTENT_TYPE_LATEST
from fastapi import Response
from functools import wraps
import time

# Define metrics
request_count = Counter(
    'api_requests_total',
    'Total API requests',
    ['method', 'endpoint', 'status']
)

request_duration = Histogram(
    'api_request_duration_seconds',
    'API request duration',
    ['method', 'endpoint']
)

active_users = Gauge(
    'active_users',
    'Number of active users'
)

videos_generated = Counter(
    'videos_generated_total',
    'Total videos generated',
    ['channel', 'style']
)

video_generation_duration = Histogram(
    'video_generation_duration_seconds',
    'Video generation duration',
    ['style'],
    buckets=(30, 60, 120, 240, 480, 960, float('inf'))
)

api_cost = Counter(
    'api_cost_dollars',
    'API costs in dollars',
    ['service']
)

youtube_quota = Gauge(
    'youtube_quota_used',
    'YouTube API quota used',
    ['account']
)

# Decorator for timing functions
def track_time(metric):
    def decorator(func):
        @wraps(func)
        async def wrapper(*args, **kwargs):
            start = time.time()
            try:
                result = await func(*args, **kwargs)
                return result
            finally:
                duration = time.time() - start
                metric.observe(duration)
        return wrapper
    return decorator

# Middleware for request tracking
class MetricsMiddleware:
    def __init__(self, app):
        self.app = app
    
    async def __call__(self, request, call_next):
        start_time = time.time()
        
        response = await call_next(request)
        
        duration = time.time() - start_time
        request_count.labels(
            method=request.method,
            endpoint=request.url.path,
            status=response.status_code
        ).inc()
        
        request_duration.labels(
            method=request.method,
            endpoint=request.url.path
        ).observe(duration)
        
        return response

# Metrics endpoint
async def metrics_endpoint():
    return Response(content=generate_latest(), media_type=CONTENT_TYPE_LATEST)
```

### Logging Configuration

```python
# app/core/logging_config.py
import logging
import json
from pythonjsonlogger import jsonlogger
from logging.handlers import RotatingFileHandler
import sys

def setup_logging(app_name: str = "ytempire"):
    """Configure structured logging"""
    
    # Create logger
    logger = logging.getLogger(app_name)
    logger.setLevel(logging.INFO)
    
    # JSON formatter
    json_formatter = jsonlogger.JsonFormatter(
        fmt='%(asctime)s %(levelname)s %(name)s %(funcName)s %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    
    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(json_formatter)
    logger.addHandler(console_handler)
    
    # File handler with rotation
    file_handler = RotatingFileHandler(
        '/var/log/ytempire/app.log',
        maxBytes=100 * 1024 * 1024,  # 100MB
        backupCount=10
    )
    file_handler.setFormatter(json_formatter)
    logger.addHandler(file_handler)
    
    # Error file handler
    error_handler = RotatingFileHandler(
        '/var/log/ytempire/error.log',
        maxBytes=50 * 1024 * 1024,  # 50MB
        backupCount=5
    )
    error_handler.setLevel(logging.ERROR)
    error_handler.setFormatter(json_formatter)
    logger.addHandler(error_handler)
    
    return logger

# Usage in application
logger = setup_logging()

# Structured logging examples
logger.info("API request received", extra={
    "user_id": "123",
    "endpoint": "/api/v1/videos",
    "method": "POST",
    "ip": "192.168.1.1"
})

logger.error("Video generation failed", extra={
    "video_id": "456",
    "error": "OpenAI rate limit",
    "retry_count": 3
})
```

### Monitoring Dashboards

```yaml
# Grafana Dashboard Configuration
dashboards:
  - name: API Performance
    panels:
      - Request Rate (req/s)
      - Response Time (p50, p95, p99)
      - Error Rate (%)
      - Active Connections
      
  - name: Video Generation
    panels:
      - Videos Generated (hourly)
      - Generation Time Distribution
      - Success/Failure Rate
      - Cost per Video
      
  - name: External Services
    panels:
      - YouTube Quota Usage
      - OpenAI API Calls
      - ElevenLabs Usage
      - Service Error Rates
      
  - name: System Resources
    panels:
      - CPU Usage
      - Memory Usage
      - Disk I/O
      - Network Traffic
      
  - name: Business Metrics
    panels:
      - Active Users
      - Revenue (daily/monthly)
      - Videos per User
      - Cost Analysis
```

---

## 9.3 Performance Optimization

### Caching Strategy

```python
# app/core/cache.py
import redis
import json
import hashlib
from typing import Optional, Any
from functools import wraps
from datetime import timedelta

redis_client = redis.from_url(settings.REDIS_URL)

class CacheService:
    def __init__(self):
        self.ttl = {
            'user_profile': 3600,        # 1 hour
            'channel_list': 300,         # 5 minutes
            'video_details': 600,        # 10 minutes
            'analytics': 60,             # 1 minute
            'api_response': 300          # 5 minutes
        }
    
    def _generate_key(self, prefix: str, *args, **kwargs) -> str:
        """Generate cache key from arguments"""
        key_data = f"{prefix}:{args}:{sorted(kwargs.items())}"
        return hashlib.md5(key_data.encode()).hexdigest()
    
    async def get(self, key: str) -> Optional[Any]:
        """Get value from cache"""
        value = redis_client.get(key)
        if value:
            return json.loads(value)
        return None
    
    async def set(self, key: str, value: Any, ttl: int = None):
        """Set value in cache"""
        redis_client.setex(
            key,
            ttl or self.ttl.get('api_response', 300),
            json.dumps(value)
        )
    
    async def invalidate(self, pattern: str):
        """Invalidate cache by pattern"""
        for key in redis_client.scan_iter(match=pattern):
            redis_client.delete(key)

cache_service = CacheService()

# Cache decorator
def cache_result(ttl: int = 300, prefix: str = "api"):
    def decorator(func):
        @wraps(func)
        async def wrapper(*args, **kwargs):
            # Generate cache key
            cache_key = cache_service._generate_key(prefix, *args, **kwargs)
            
            # Try to get from cache
            cached = await cache_service.get(cache_key)
            if cached:
                return cached
            
            # Execute function
            result = await func(*args, **kwargs)
            
            # Store in cache
            await cache_service.set(cache_key, result, ttl)
            
            return result
        return wrapper
    return decorator
```

### Database Optimization

```python
# app/core/database_optimization.py
from sqlalchemy import event, create_engine
from sqlalchemy.engine import Engine
from sqlalchemy.pool import QueuePool
from sqlalchemy.orm import Session
from typing import List, Any
import time
import logging

logger = logging.getLogger(__name__)

# Connection pool configuration
engine = create_engine(
    settings.DATABASE_URL,
    poolclass=QueuePool,
    pool_size=20,
    max_overflow=40,
    pool_pre_ping=True,
    pool_recycle=3600,
    echo_pool=settings.DEBUG
)

# Query performance monitoring
@event.listens_for(Engine, "before_cursor_execute")
def before_cursor_execute(conn, cursor, statement, parameters, context, executemany):
    conn.info.setdefault('query_start_time', []).append(time.time())

@event.listens_for(Engine, "after_cursor_execute")
def after_cursor_execute(conn, cursor, statement, parameters, context, executemany):
    total = time.time() - conn.info['query_start_time'].pop(-1)
    if total > 0.5:  # Log slow queries
        logger.warning(f"Slow query ({total:.2f}s): {statement[:100]}")

# Query optimization helpers
class QueryOptimizer:
    @staticmethod
    def batch_insert(db: Session, objects: List[Any], batch_size: int = 1000):
        """Batch insert for better performance"""
        for i in range(0, len(objects), batch_size):
            batch = objects[i:i + batch_size]
            db.bulk_insert_mappings(type(batch[0]), [obj.__dict__ for obj in batch])
            db.commit()
    
    @staticmethod
    def use_read_replica(query):
        """Route read queries to replica"""
        return query.execution_options(
            synchronize_session=False,
            bind=read_replica_engine
        )
```

### API Response Optimization

```python
# app/core/response_optimization.py
from fastapi import Response
import gzip
import json

class CompressionMiddleware:
    """Compress responses for better performance"""
    
    def __init__(self, app):
        self.app = app
    
    async def __call__(self, request, call_next):
        response = await call_next(request)
        
        # Check if client accepts gzip
        accept_encoding = request.headers.get('accept-encoding', '')
        if 'gzip' not in accept_encoding:
            return response
        
        # Compress response for successful requests
        if response.status_code == 200:
            # Collect response body
            body = b''
            async for chunk in response.body_iterator:
                body += chunk
            
            # Compress body
            compressed = gzip.compress(body)
            
            # Only use compression if it's worth it (>10% reduction)
            if len(compressed) < len(body) * 0.9:
                return Response(
                    content=compressed,
                    media_type=response.media_type,
                    status_code=response.status_code,
                    headers={
                        **dict(response.headers),
                        'content-encoding': 'gzip',
                        'content-length': str(len(compressed))
                    }
                )
        
        return response
```

---

## 9.4 Disaster Recovery

### Backup Strategy

```yaml
Backup Schedule:
  Database:
    - Full backup: Daily at 2 AM
    - Incremental: Every 6 hours
    - Transaction logs: Continuous
    - Retention: 30 days
  
  Files:
    - User uploads: Daily
    - Generated videos: Weekly
    - Logs: Monthly
    - Retention: 90 days
  
  Configuration:
    - Git repository: Every change
    - Secrets: Encrypted daily
    - Environment: Weekly snapshot

Recovery Targets:
  RPO (Recovery Point Objective): 1 hour
  RTO (Recovery Time Objective): 4 hours
  Data Loss Tolerance: <1 hour of data
```

### Backup Scripts

```bash
#!/bin/bash
# scripts/backup.sh

# Configuration
BACKUP_DIR="/backups"
DATE=$(date +%Y%m%d-%H%M%S)
RETENTION_DAYS=30

# Database backup
backup_database() {
    echo "Backing up database..."
    
    # Create backup
    pg_dump $DATABASE_URL | gzip > "$BACKUP_DIR/db/ytempire-$DATE.sql.gz"
    
    # Upload to S3
    aws s3 cp "$BACKUP_DIR/db/ytempire-$DATE.sql.gz" \
        s3://ytempire-backups/database/ \
        --storage-class GLACIER
    
    # Clean old backups
    find $BACKUP_DIR/db -name "*.sql.gz" -mtime +$RETENTION_DAYS -delete
}

# File backup
backup_files() {
    echo "Backing up files..."
    
    # Create tar archive
    tar -czf "$BACKUP_DIR/files/files-$DATE.tar.gz" \
        /storage/videos \
        /storage/thumbnails \
        /storage/scripts
    
    # Upload to S3
    aws s3 sync /storage/ s3://ytempire-backups/files/ \
        --exclude "*.tmp" \
        --storage-class GLACIER_IR
}

# Configuration backup
backup_config() {
    echo "Backing up configuration..."
    
    # Encrypt sensitive configs
    tar -czf - /etc/ytempire/ | \
        openssl enc -aes-256-cbc -salt -pass pass:$BACKUP_PASSWORD \
        > "$BACKUP_DIR/config/config-$DATE.tar.gz.enc"
}

# Main execution
main() {
    backup_database
    backup_files
    backup_config
    
    echo "Backup completed successfully"
}

main
```

### Recovery Procedures

```bash
#!/bin/bash
# scripts/disaster_recovery.sh

# Disaster Recovery Runbook

# 1. System Recovery
recover_system() {
    echo "=== System Recovery ==="
    
    # Provision new server (if needed)
    # This assumes server is available
    
    # Install dependencies
    apt-get update
    apt-get install -y docker docker-compose postgresql-client
}

# 2. Database Recovery
recover_database() {
    echo "=== Database Recovery ==="
    
    # Find latest backup
    LATEST_BACKUP=$(aws s3 ls s3://ytempire-backups/database/ | tail -1 | awk '{print $4}')
    
    # Download backup
    aws s3 cp "s3://ytempire-backups/database/$LATEST_BACKUP" /tmp/
    
    # Restore database
    gunzip < "/tmp/$LATEST_BACKUP" | psql $DATABASE_URL
    
    # Verify integrity
    psql $DATABASE_URL -c "SELECT COUNT(*) FROM users;"
}

# 3. Application Recovery
recover_application() {
    echo "=== Application Recovery ==="
    
    # Pull latest code
    git clone https://github.com/ytempire/backend.git /opt/ytempire
    cd /opt/ytempire
    
    # Restore configuration
    aws s3 cp s3://ytempire-backups/config/latest.tar.gz.enc /tmp/
    openssl enc -d -aes-256-cbc -pass pass:$BACKUP_PASSWORD \
        -in /tmp/latest.tar.gz.enc | tar -xzf - -C /
    
    # Start services
    docker-compose up -d
}

# 4. Verification
verify_recovery() {
    echo "=== Verification ==="
    
    # Health checks
    curl -f http://localhost:8000/health || exit 1
    
    # Database connectivity
    psql $DATABASE_URL -c "SELECT 1;" || exit 1
    
    # Redis connectivity
    redis-cli ping || exit 1
    
    echo "Recovery completed successfully!"
}

# Main recovery process
main() {
    recover_system
    recover_database
    recover_application
    verify_recovery
}

main
```

---

## 9.5 Maintenance Procedures

### Regular Maintenance Tasks

```yaml
Daily:
  - Check system health metrics
  - Review error logs
  - Monitor API costs
  - Verify backup completion
  - Check YouTube quota usage

Weekly:
  - Database vacuum and analyze
  - Clean temporary files
  - Update dependencies
  - Security scan
  - Performance review

Monthly:
  - Full system backup test
  - Disaster recovery drill
  - Security audit
  - Cost optimization review
  - Capacity planning

Quarterly:
  - Major version updates
  - Infrastructure review
  - Security penetration test
  - Architecture review
```

### Maintenance Scripts

```python
# app/maintenance/tasks.py
import asyncio
from datetime import datetime, timedelta
from sqlalchemy import text
import logging
import shutil
import os

class MaintenanceTasks:
    def __init__(self, db, logger):
        self.db = db
        self.logger = logger
    
    async def daily_maintenance(self):
        """Daily maintenance tasks"""
        self.logger.info("Starting daily maintenance")
        
        # Clean old sessions
        await self._clean_sessions()
        
        # Reset YouTube quotas
        await self._reset_youtube_quotas()
        
        # Archive old logs
        await self._archive_logs()
        
        # Generate daily report
        await self._generate_daily_report()
        
        self.logger.info("Daily maintenance completed")
    
    async def _clean_sessions(self):
        """Clean expired sessions"""
        cutoff = datetime.utcnow() - timedelta(days=7)
        result = await self.db.execute(
            text("DELETE FROM sessions WHERE created_at < :cutoff"),
            {"cutoff": cutoff}
        )
        self.logger.info(f"Cleaned {result.rowcount} expired sessions")
    
    async def _reset_youtube_quotas(self):
        """Reset YouTube account quotas"""
        await self.db.execute(
            text("""
                UPDATE youtube_accounts 
                SET quota_used_today = 0, 
                    uploads_today = 0,
                    last_quota_reset = NOW()
                WHERE last_quota_reset < CURRENT_DATE
            """)
        )
        await self.db.commit()
    
    async def _archive_logs(self):
        """Archive old log files"""
        log_dir = "/var/log/ytempire"
        archive_dir = "/var/log/ytempire/archive"
        
        # Create archive directory if it doesn't exist
        os.makedirs(archive_dir, exist_ok=True)
        
        for filename in os.listdir(log_dir):
            if filename.endswith(".log.1"):
                src = os.path.join(log_dir, filename)
                dst = os.path.join(archive_dir, f"{filename}.{datetime.now().strftime('%Y%m%d')}")
                shutil.move(src, dst)
                
                # Compress archived log
                os.system(f"gzip {dst}")
    
    async def _generate_daily_report(self):
        """Generate daily operations report"""
        report = {
            "date": datetime.utcnow().date().isoformat(),
            "videos_generated": await self._get_daily_videos(),
            "total_cost": await self._get_daily_cost(),
            "active_users": await self._get_active_users(),
            "error_count": await self._get_error_count(),
            "youtube_quota_usage": await self._get_quota_usage()
        }
        
        # Send report via email or Slack
        await self._send_report(report)
    
    async def _get_daily_videos(self):
        """Get count of videos generated today"""
        result = await self.db.execute(
            text("""
                SELECT COUNT(*) as count 
                FROM videos 
                WHERE DATE(created_at) = CURRENT_DATE
            """)
        )
        return result.scalar()
    
    async def _get_daily_cost(self):
        """Get total cost for today"""
        result = await self.db.execute(
            text("""
                SELECT SUM(amount) as total 
                FROM video_costs 
                WHERE DATE(created_at) = CURRENT_DATE
            """)
        )
        return float(result.scalar() or 0)
    
    async def _get_active_users(self):
        """Get count of active users today"""
        result = await self.db.execute(
            text("""
                SELECT COUNT(DISTINCT user_id) as count 
                FROM user_activity 
                WHERE DATE(activity_time) = CURRENT_DATE
            """)
        )
        return result.scalar()
    
    async def _get_error_count(self):
        """Get error count for today"""
        # This would read from log files or error tracking table
        return 0
    
    async def _get_quota_usage(self):
        """Get YouTube quota usage across all accounts"""
        result = await self.db.execute(
            text("""
                SELECT 
                    AVG(quota_used_today / quota_limit * 100) as avg_usage,
                    MAX(quota_used_today / quota_limit * 100) as max_usage
                FROM youtube_accounts
                WHERE is_active = true
            """)
        )
        row = result.fetchone()
        return {
            "average_usage": round(row.avg_usage, 2),
            "max_usage": round(row.max_usage, 2)
        }
    
    async def _send_report(self, report):
        """Send daily report to stakeholders"""
        # Implementation for sending report via email or Slack
        self.logger.info(f"Daily report: {report}")

# Usage
maintenance = MaintenanceTasks(db, logger)
asyncio.create_task(maintenance.daily_maintenance())
```

### Database Maintenance

```sql
-- Database maintenance procedures

-- Vacuum and analyze all tables
VACUUM ANALYZE;

-- Reindex for better performance
REINDEX DATABASE ytempire;

-- Update table statistics
ANALYZE users, channels, videos, video_costs;

-- Clean up old data
DELETE FROM audit_logs WHERE created_at < NOW() - INTERVAL '90 days';
DELETE FROM api_logs WHERE created_at < NOW() - INTERVAL '30 days';
DELETE FROM sessions WHERE created_at < NOW() - INTERVAL '7 days';

-- Check table health
SELECT 
    schemaname,
    tablename,
    n_dead_tup,
    n_live_tup,
    round(n_dead_tup::numeric / NULLIF(n_live_tup, 0), 4) AS dead_ratio
FROM pg_stat_user_tables
WHERE n_dead_tup > 1000
ORDER BY dead_ratio DESC;

-- Monitor table sizes
SELECT 
    nspname || '.' || relname AS "relation",
    pg_size_pretty(pg_total_relation_size(C.oid)) AS "total_size"
FROM pg_class C
LEFT JOIN pg_namespace N ON (N.oid = C.relnamespace)
WHERE nspname NOT IN ('pg_catalog', 'information_schema')
    AND C.relkind <> 'i'
    AND nspname !~ '^pg_toast'
ORDER BY pg_total_relation_size(C.oid) DESC
LIMIT 20;

-- Check for unused indexes
SELECT 
    schemaname,
    tablename,
    indexname,
    idx_scan,
    idx_tup_read,
    idx_tup_fetch,
    pg_size_pretty(pg_relation_size(indexrelid)) AS index_size
FROM pg_stat_user_indexes
WHERE idx_scan = 0
ORDER BY pg_relation_size(indexrelid) DESC;

-- Lock monitoring
SELECT 
    pid,
    age(clock_timestamp(), query_start) AS duration,
    query,
    state
FROM pg_stat_activity
WHERE state != 'idle'
ORDER BY duration DESC;
```

### System Maintenance

```bash
#!/bin/bash
# scripts/system_maintenance.sh

# System maintenance script

# Clean Docker resources
clean_docker() {
    echo "Cleaning Docker resources..."
    
    # Remove stopped containers
    docker container prune -f
    
    # Remove unused images
    docker image prune -a -f
    
    # Remove unused volumes
    docker volume prune -f
    
    # Remove unused networks
    docker network prune -f
    
    # Clean build cache
    docker builder prune -f
}

# Clean temporary files
clean_temp_files() {
    echo "Cleaning temporary files..."
    
    # Clean application temp files
    find /tmp -name "*.tmp" -mtime +1 -delete
    
    # Clean old video processing files
    find /storage/temp -name "*" -mtime +7 -delete
    
    # Clean old log files
    find /var/log/ytempire -name "*.log.*" -mtime +30 -delete
}

# Update system packages
update_system() {
    echo "Updating system packages..."
    
    apt-get update
    apt-get upgrade -y
    apt-get autoremove -y
    apt-get autoclean
}

# Check disk space
check_disk_space() {
    echo "Checking disk space..."
    
    df -h | grep -E '^/dev/'
    
    # Alert if any partition is over 80%
    df -h | grep -E '^/dev/' | awk '{print $5 " " $6}' | while read usage mount; do
        usage_num=${usage%\%}
        if [ $usage_num -gt 80 ]; then
            echo "WARNING: $mount is at $usage capacity"
            # Send alert
        fi
    done
}

# Main execution
main() {
    echo "Starting system maintenance..."
    
    clean_docker
    clean_temp_files
    update_system
    check_disk_space
    
    echo "System maintenance completed"
}

main
```

---

## Emergency Procedures

### Incident Response

```yaml
Incident Levels:
  P1 - Critical:
    - Complete service outage
    - Data breach
    - Payment system failure
    Response: Immediate, all hands
    
  P2 - High:
    - Partial service degradation
    - YouTube API failure
    - High error rate (>10%)
    Response: Within 30 minutes
    
  P3 - Medium:
    - Performance degradation
    - Non-critical service failure
    Response: Within 2 hours
    
  P4 - Low:
    - Minor bugs
    - Cosmetic issues
    Response: Next business day

Response Process:
  1. Detect - Monitoring alert
  2. Assess - Determine severity
  3. Communicate - Notify stakeholders
  4. Mitigate - Temporary fix
  5. Resolve - Permanent solution
  6. Review - Post-mortem
```

### Emergency Response Scripts

```python
# app/emergency/incident_response.py
import asyncio
from datetime import datetime
from typing import Dict, Any
import logging

class IncidentResponse:
    def __init__(self):
        self.logger = logging.getLogger(__name__)
    
    async def handle_outage(self, service: str):
        """Handle service outage"""
        self.logger.critical(f"OUTAGE DETECTED: {service}")
        
        # Step 1: Assess impact
        impact = await self._assess_impact(service)
        
        # Step 2: Notify team
        await self._notify_team(service, impact)
        
        # Step 3: Implement mitigation
        if service == "youtube":
            await self._handle_youtube_outage()
        elif service == "openai":
            await self._handle_openai_outage()
        elif service == "database":
            await self._handle_database_outage()
        
        # Step 4: Monitor recovery
        await self._monitor_recovery(service)
    
    async def _assess_impact(self, service: str) -> Dict[str, Any]:
        """Assess the impact of the outage"""
        return {
            "service": service,
            "users_affected": await self._count_affected_users(),
            "videos_blocked": await self._count_blocked_videos(),
            "revenue_impact": await self._calculate_revenue_impact()
        }
    
    async def _notify_team(self, service: str, impact: Dict):
        """Notify team of incident"""
        message = f"""
        🚨 INCIDENT ALERT 🚨
        Service: {service}
        Time: {datetime.utcnow().isoformat()}
        Impact:
        - Users affected: {impact['users_affected']}
        - Videos blocked: {impact['videos_blocked']}
        - Revenue impact: ${impact['revenue_impact']:.2f}
        
        Please join incident channel immediately.
        """
        
        # Send notifications via multiple channels
        # Slack, Email, SMS, PagerDuty
        self.logger.critical(message)
    
    async def _handle_youtube_outage(self):
        """Handle YouTube API outage"""
        # Switch to reserve accounts
        # Queue videos for later upload
        # Notify users of delay
        pass
    
    async def _handle_openai_outage(self):
        """Handle OpenAI API outage"""
        # Switch to fallback model (GPT-3.5 or Claude)
        # Use cached scripts if available
        # Reduce video generation rate
        pass
    
    async def _handle_database_outage(self):
        """Handle database outage"""
        # Switch to read replica if available
        # Enable emergency cache mode
        # Queue writes for later processing
        pass
    
    async def _monitor_recovery(self, service: str):
        """Monitor service recovery"""
        recovered = False
        attempts = 0
        
        while not recovered and attempts < 60:  # Try for 1 hour
            await asyncio.sleep(60)  # Check every minute
            
            if await self._check_service_health(service):
                recovered = True
                self.logger.info(f"Service {service} has recovered")
                await self._notify_recovery(service)
            
            attempts += 1
    
    async def _check_service_health(self, service: str) -> bool:
        """Check if service is healthy"""
        # Implementation for health checking
        return False
    
    async def _notify_recovery(self, service: str):
        """Notify team of recovery"""
        message = f"""
        ✅ INCIDENT RESOLVED
        Service: {service}
        Recovery Time: {datetime.utcnow().isoformat()}
        
        Service has been restored. Please verify functionality.
        """
        self.logger.info(message)

# Usage
incident_handler = IncidentResponse()
```

### Emergency Contacts

```yaml
contacts:
  on_call:
    primary: "+1-555-0100"
    secondary: "+1-555-0101"
  
  escalation:
    backend_lead: "+1-555-0102"
    cto: "+1-555-0103"
    ceo: "+1-555-0104"
  
  vendors:
    aws_support: "https://console.aws.amazon.com/support"
    google_cloud: "https://cloud.google.com/support"
    stripe: "https://support.stripe.com"
    openai: "https://help.openai.com"
  
  internal:
    slack_channel: "#incidents"
    war_room: "https://zoom.us/j/emergency"
    status_page: "https://status.ytempire.com"
```

---

## Performance Benchmarks

### Load Testing

```python
# tests/load_test.py
import asyncio
import aiohttp
import time
from typing import List

class LoadTest:
    def __init__(self, base_url: str):
        self.base_url = base_url
        self.results = []
    
    async def run_test(self, concurrent_users: int, duration: int):
        """Run load test"""
        print(f"Starting load test: {concurrent_users} users for {duration} seconds")
        
        start_time = time.time()
        tasks = []
        
        for i in range(concurrent_users):
            task = asyncio.create_task(self._simulate_user(i, start_time, duration))
            tasks.append(task)
        
        await asyncio.gather(*tasks)
        
        # Calculate results
        self._calculate_results()
    
    async def _simulate_user(self, user_id: int, start_time: float, duration: int):
        """Simulate a single user"""
        async with aiohttp.ClientSession() as session:
            while time.time() - start_time < duration:
                # Simulate various API calls
                await self._make_request(session, "GET", "/api/v1/channels")
                await asyncio.sleep(1)
                await self._make_request(session, "GET", "/api/v1/videos")
                await asyncio.sleep(2)
                await self._make_request(session, "POST", "/api/v1/videos/generate")
                await asyncio.sleep(5)
    
    async def _make_request(self, session, method: str, path: str):
        """Make a single request and record metrics"""
        url = f"{self.base_url}{path}"
        start = time.time()
        
        try:
            async with session.request(method, url) as response:
                duration = time.time() - start
                self.results.append({
                    "method": method,
                    "path": path,
                    "status": response.status,
                    "duration": duration,
                    "timestamp": time.time()
                })
        except Exception as e:
            self.results.append({
                "method": method,
                "path": path,
                "status": 0,
                "duration": time.time() - start,
                "error": str(e),
                "timestamp": time.time()
            })
    
    def _calculate_results(self):
        """Calculate and display results"""
        total_requests = len(self.results)
        successful = sum(1 for r in self.results if 200 <= r.get("status", 0) < 300)
        failed = total_requests - successful
        
        durations = [r["duration"] for r in self.results if r.get("status", 0) == 200]
        if durations:
            durations.sort()
            p50 = durations[len(durations) // 2]
            p95 = durations[int(len(durations) * 0.95)]
            p99 = durations[int(len(durations) * 0.99)]
            
            print(f"""
            Load Test Results:
            ==================
            Total Requests: {total_requests}
            Successful: {successful}
            Failed: {failed}
            Success Rate: {successful/total_requests*100:.2f}%
            
            Response Times:
            P50: {p50*1000:.2f}ms
            P95: {p95*1000:.2f}ms
            P99: {p99*1000:.2f}ms
            """)

# Run load test
async def main():
    load_test = LoadTest("http://localhost:8000")
    await load_test.run_test(concurrent_users=50, duration=60)

if __name__ == "__main__":
    asyncio.run(main())
```

---

## Document Control

- **Version**: 2.0
- **Last Updated**: January 2025
- **Next Review**: February 2025
- **Owner**: Platform Ops Lead
- **Approved By**: CTO/Technical Director

---

## Navigation

- [← Previous: External Integrations](./8-external-integrations.md)
- [→ Next: Team Collaboration](./10-team-collaboration.md)