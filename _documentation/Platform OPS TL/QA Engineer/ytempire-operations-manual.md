# YTEMPIRE Operations Manual

## 7.1 Deployment Procedures

### Deployment Strategy

#### Deployment Approach
- **Method**: Blue-Green deployment
- **Frequency**: Once daily maximum (MVP)
- **Window**: 10 PM - 2 AM PST
- **Duration**: <10 minutes
- **Rollback Time**: <5 minutes

### Pre-Deployment Checklist

```markdown
## Pre-Deployment Verification

### Code Readiness ✓
- [ ] All tests passing (>95% pass rate)
- [ ] Code review approved
- [ ] No critical bugs open
- [ ] Documentation updated
- [ ] Release notes prepared

### Infrastructure Check ✓
- [ ] Server resources available (CPU <70%, Memory <80%)
- [ ] Disk space sufficient (>20% free)
- [ ] Backup completed successfully
- [ ] Database migrations tested
- [ ] Network connectivity verified

### Communication ✓
- [ ] Team notified via Slack
- [ ] Maintenance window announced (if needed)
- [ ] On-call engineer confirmed
- [ ] Rollback plan reviewed
```

### Deployment Process

#### Step-by-Step Deployment

```bash
#!/bin/bash
# deploy.sh - Main deployment script

set -e  # Exit on error

# Configuration
VERSION=$1
ENVIRONMENT=${2:-production}
ROLLBACK_POINT=$(date +%Y%m%d_%H%M%S)

echo "==================================="
echo "YTEMPIRE Deployment v${VERSION}"
echo "Environment: ${ENVIRONMENT}"
echo "Timestamp: ${ROLLBACK_POINT}"
echo "==================================="

# Step 1: Pre-deployment backup
echo "[1/8] Creating backup..."
./scripts/backup.sh ${ROLLBACK_POINT}

# Step 2: Pull latest code
echo "[2/8] Pulling latest code..."
git fetch --all
git checkout tags/v${VERSION}

# Step 3: Build containers
echo "[3/8] Building Docker containers..."
docker-compose -f docker-compose.${ENVIRONMENT}.yml build

# Step 4: Run database migrations
echo "[4/8] Running database migrations..."
docker-compose run --rm backend python manage.py migrate

# Step 5: Start new containers (Blue-Green)
echo "[5/8] Starting new containers..."
docker-compose -f docker-compose.${ENVIRONMENT}.yml up -d --scale app=2

# Step 6: Health check
echo "[6/8] Running health checks..."
./scripts/health-check.sh || {
    echo "Health check failed! Rolling back..."
    ./scripts/rollback.sh ${ROLLBACK_POINT}
    exit 1
}

# Step 7: Switch traffic
echo "[7/8] Switching traffic to new version..."
docker-compose -f docker-compose.${ENVIRONMENT}.yml stop app_old
docker-compose -f docker-compose.${ENVIRONMENT}.yml rm -f app_old

# Step 8: Clear cache
echo "[8/8] Clearing cache..."
docker-compose exec redis redis-cli FLUSHALL

echo "==================================="
echo "Deployment completed successfully!"
echo "Version ${VERSION} is now live"
echo "==================================="

# Send notification
./scripts/notify-deployment.sh ${VERSION} "success"
```

#### Docker Compose Configuration

```yaml
# docker-compose.production.yml
version: '3.8'

services:
  nginx:
    image: nginx:alpine
    ports:
      - "80:80"
      - "443:443"
    volumes:
      - ./nginx/nginx.conf:/etc/nginx/nginx.conf
      - ./ssl:/etc/nginx/ssl
    depends_on:
      - app
    restart: unless-stopped

  app:
    image: ytempire/backend:${VERSION}
    environment:
      - DATABASE_URL=${DATABASE_URL}
      - REDIS_URL=redis://redis:6379
      - ENVIRONMENT=production
    depends_on:
      - postgres
      - redis
    restart: unless-stopped
    healthcheck:
      test: ["CMD", "curl", "-f", "http://localhost:8000/health"]
      interval: 30s
      timeout: 10s
      retries: 3

  postgres:
    image: postgres:15
    volumes:
      - postgres_data:/var/lib/postgresql/data
    environment:
      - POSTGRES_DB=ytempire
      - POSTGRES_USER=${DB_USER}
      - POSTGRES_PASSWORD=${DB_PASSWORD}
    restart: unless-stopped

  redis:
    image: redis:7-alpine
    command: redis-server --appendonly yes
    volumes:
      - redis_data:/data
    restart: unless-stopped

  celery:
    image: ytempire/backend:${VERSION}
    command: celery -A app.celery worker -l info
    environment:
      - DATABASE_URL=${DATABASE_URL}
      - REDIS_URL=redis://redis:6379
    depends_on:
      - redis
      - postgres
    restart: unless-stopped

volumes:
  postgres_data:
  redis_data:
```

### Post-Deployment Verification

```bash
#!/bin/bash
# post-deployment-check.sh

echo "Running post-deployment verification..."

# Check service health
for service in nginx app postgres redis celery; do
    status=$(docker-compose ps ${service} | grep Up)
    if [ -z "$status" ]; then
        echo "ERROR: Service ${service} is not running!"
        exit 1
    fi
    echo "✓ ${service} is healthy"
done

# Check API endpoints
endpoints=(
    "/health"
    "/api/v1/status"
    "/api/v1/channels"
)

for endpoint in "${endpoints[@]}"; do
    response=$(curl -s -o /dev/null -w "%{http_code}" http://localhost:8000${endpoint})
    if [ $response -ne 200 ]; then
        echo "ERROR: Endpoint ${endpoint} returned ${response}"
        exit 1
    fi
    echo "✓ ${endpoint} is responding"
done

# Check database connectivity
docker-compose exec postgres psql -U ${DB_USER} -d ytempire -c "SELECT 1" > /dev/null 2>&1
echo "✓ Database is accessible"

# Check Redis
docker-compose exec redis redis-cli ping > /dev/null 2>&1
echo "✓ Redis is responding"

echo "Post-deployment verification completed successfully!"
```

## 7.2 Monitoring & Alerting

### Monitoring Architecture

```yaml
Monitoring Stack:
  Metrics Collection:
    Tool: Prometheus
    Interval: 30 seconds
    Retention: 30 days
    
  Visualization:
    Tool: Grafana
    Dashboards:
      - System Overview
      - Application Performance
      - Business Metrics
      - Cost Tracking
    
  Alerting:
    Primary: Prometheus Alertmanager
    Channels:
      - Slack (#alerts)
      - Email (on-call@ytempire.com)
      - PagerDuty (critical only)
    
  Logging:
    Collection: Docker logs
    Rotation: Daily
    Retention: 7 days local, 30 days archive
```

### Key Metrics to Monitor

#### System Metrics
```yaml
CPU Usage:
  Warning: >70%
  Critical: >85%
  Action: Scale or optimize

Memory Usage:
  Warning: >80%
  Critical: >90%
  Action: Investigate leaks

Disk Usage:
  Warning: >70%
  Critical: >85%
  Action: Clean up or expand

Network:
  Bandwidth: <800 Mbps warning
  Packet Loss: >1% warning
  Latency: >100ms warning
```

#### Application Metrics
```yaml
API Performance:
  Response Time:
    p50: <200ms
    p95: <500ms
    p99: <1000ms
  
  Error Rate:
    Warning: >1%
    Critical: >5%
  
  Request Rate:
    Monitor for anomalies
    Alert on >2x normal

Queue Metrics:
  Queue Depth:
    Warning: >50 jobs
    Critical: >100 jobs
  
  Processing Time:
    Warning: >12 min/video
    Critical: >15 min/video
  
  Failed Jobs:
    Warning: >5%
    Critical: >10%
```

#### Business Metrics
```yaml
Video Generation:
  Success Rate:
    Warning: <90%
    Critical: <80%
  
  Cost per Video:
    Warning: >$2.50
    Critical: >$3.00
  
  Daily Volume:
    Track against capacity

User Activity:
  Active Users:
    Track daily/weekly/monthly
  
  New Registrations:
    Monitor for anomalies
  
  Churn Indicators:
    Decreased activity
    Failed payments
```

### Alert Configuration

```yaml
# prometheus-alerts.yml
groups:
  - name: system_alerts
    rules:
      - alert: HighCPUUsage
        expr: node_cpu_usage > 85
        for: 5m
        labels:
          severity: critical
        annotations:
          summary: "High CPU usage detected"
          description: "CPU usage is above 85% for 5 minutes"
      
      - alert: LowDiskSpace
        expr: node_filesystem_free_bytes / node_filesystem_size_bytes < 0.15
        for: 10m
        labels:
          severity: warning
        annotations:
          summary: "Low disk space"
          description: "Less than 15% disk space remaining"
  
  - name: application_alerts
    rules:
      - alert: HighAPIResponseTime
        expr: http_request_duration_seconds{quantile="0.95"} > 0.5
        for: 5m
        labels:
          severity: warning
        annotations:
          summary: "API response time degraded"
          description: "95th percentile response time above 500ms"
      
      - alert: VideoGenerationFailure
        expr: rate(video_generation_failures[5m]) > 0.1
        for: 5m
        labels:
          severity: critical
        annotations:
          summary: "High video generation failure rate"
          description: "More than 10% of video generations failing"
```

### Monitoring Dashboards

#### System Overview Dashboard
- Server health status
- Resource utilization graphs
- Network traffic
- Disk I/O metrics
- Container status

#### Application Performance Dashboard
- API response times
- Request rates
- Error rates
- Queue metrics
- Cache hit rates

#### Business Metrics Dashboard
- Active users
- Videos generated
- Revenue tracking
- Cost per video
- Channel growth

## 7.3 Disaster Recovery

### Backup Strategy

#### Backup Schedule
```yaml
Backup Configuration:
  Database:
    Type: PostgreSQL dump
    Frequency: Every 4 hours
    Retention: 
      Local: 7 days
      Remote: 30 days
    Verification: Daily restore test
  
  File Storage:
    Type: Incremental rsync
    Frequency: Daily
    Retention:
      Local: 3 days
      Remote: 14 days
  
  Configuration:
    Type: Git repository
    Frequency: On change
    Retention: Unlimited
  
  Secrets:
    Type: Encrypted backup
    Frequency: On change
    Storage: Secure vault
```

#### Backup Implementation

```bash
#!/bin/bash
# backup.sh - Automated backup script

BACKUP_DIR="/backup"
DATE=$(date +%Y%m%d_%H%M%S)
S3_BUCKET="s3://ytempire-backups"

# Database backup
echo "Backing up database..."
docker-compose exec -T postgres pg_dump -U ${DB_USER} ytempire | gzip > ${BACKUP_DIR}/db_${DATE}.sql.gz

# Verify backup
gunzip -c ${BACKUP_DIR}/db_${DATE}.sql.gz | head -n 10 > /dev/null
if [ $? -ne 0 ]; then
    echo "ERROR: Database backup verification failed!"
    exit 1
fi

# File backup
echo "Backing up files..."
rsync -av --delete /data/videos/ ${BACKUP_DIR}/videos_${DATE}/

# Configuration backup
echo "Backing up configuration..."
tar -czf ${BACKUP_DIR}/config_${DATE}.tar.gz /app/config/

# Upload to remote storage
echo "Uploading to remote storage..."
aws s3 cp ${BACKUP_DIR}/db_${DATE}.sql.gz ${S3_BUCKET}/database/
aws s3 sync ${BACKUP_DIR}/videos_${DATE}/ ${S3_BUCKET}/videos/
aws s3 cp ${BACKUP_DIR}/config_${DATE}.tar.gz ${S3_BUCKET}/config/

# Clean old local backups (keep 7 days)
find ${BACKUP_DIR} -type f -mtime +7 -delete

echo "Backup completed successfully!"
```

### Recovery Procedures

#### Recovery Time Objectives
- **RTO (Recovery Time Objective)**: <4 hours
- **RPO (Recovery Point Objective)**: <4 hours
- **Database Recovery**: <30 minutes
- **Full System Recovery**: <4 hours

#### Disaster Recovery Plan

```markdown
## Disaster Recovery Runbook

### Scenario 1: Database Corruption
1. Stop application servers
2. Identify last known good backup
3. Restore database from backup
4. Verify data integrity
5. Restart application servers
6. Run verification tests
7. Monitor for issues

### Scenario 2: Server Hardware Failure
1. Activate backup server (if available)
2. Or: Provision cloud instance
3. Restore from latest backup
4. Update DNS/networking
5. Verify all services
6. Monitor performance

### Scenario 3: Data Center Outage
1. Failover to cloud services
2. Restore from remote backups
3. Update DNS records
4. Notify users of temporary service
5. Monitor and optimize

### Scenario 4: Security Breach
1. Isolate affected systems
2. Preserve evidence
3. Reset all credentials
4. Restore from clean backup
5. Apply security patches
6. Conduct security audit
7. Notify affected users
```

#### Recovery Testing

```bash
#!/bin/bash
# disaster-recovery-test.sh

echo "Starting disaster recovery test..."

# Create test environment
docker-compose -f docker-compose.test.yml up -d

# Restore database
echo "Restoring database..."
gunzip -c /backup/db_latest.sql.gz | docker-compose exec -T postgres_test psql -U postgres

# Verify restoration
echo "Verifying data integrity..."
docker-compose exec postgres_test psql -U postgres -c "SELECT COUNT(*) FROM users;"
docker-compose exec postgres_test psql -U postgres -c "SELECT COUNT(*) FROM videos;"

# Test application
echo "Testing application..."
curl -f http://localhost:8001/health || exit 1

echo "Disaster recovery test completed successfully!"

# Cleanup
docker-compose -f docker-compose.test.yml down
```

## 7.4 Maintenance Procedures

### Routine Maintenance

#### Daily Tasks
```yaml
Morning (9:00 AM):
  - Check overnight alerts
  - Review system metrics
  - Verify backup completion
  - Check disk usage
  - Review error logs

Afternoon (2:00 PM):
  - Monitor performance metrics
  - Check queue depth
  - Review cost tracking
  - Update documentation

Evening (6:00 PM):
  - Prepare for overnight jobs
  - Review daily metrics
  - Update status dashboard
```

#### Weekly Tasks
```yaml
Monday:
  - Full system health check
  - Performance analysis
  - Capacity planning review

Wednesday:
  - Security updates check
  - Vulnerability scanning
  - Access review

Friday:
  - Backup verification test
  - Disaster recovery drill
  - Documentation update
```

#### Monthly Tasks
```yaml
First Monday:
  - Full security audit
  - Performance optimization
  - Cost optimization review
  - Capacity planning

Mid-Month:
  - Database maintenance
  - Index optimization
  - Archive old data
  - Clean up logs
```

### Database Maintenance

```sql
-- maintenance.sql - Monthly database maintenance

-- Update statistics
ANALYZE;

-- Reindex tables
REINDEX TABLE videos;
REINDEX TABLE channels;
REINDEX TABLE analytics;

-- Clean up old data
DELETE FROM analytics WHERE date < NOW() - INTERVAL '12 months';
DELETE FROM generation_jobs WHERE completed_at < NOW() - INTERVAL '30 days';

-- Vacuum to reclaim space
VACUUM FULL ANALYZE;

-- Check for bloat
SELECT 
    schemaname,
    tablename,
    pg_size_pretty(pg_total_relation_size(schemaname||'.'||tablename)) AS size,
    n_dead_tup,
    n_live_tup,
    round(n_dead_tup::numeric / NULLIF(n_live_tup, 0) * 100, 2) AS dead_ratio
FROM pg_stat_user_tables
WHERE n_dead_tup > 1000
ORDER BY dead_ratio DESC;
```

### System Updates

```bash
#!/bin/bash
# system-update.sh - System update procedure

# Create snapshot before updates
echo "Creating system snapshot..."
./scripts/create-snapshot.sh

# Update system packages
echo "Updating system packages..."
apt-get update
apt-get upgrade -y

# Update Docker
echo "Updating Docker..."
apt-get install docker-ce docker-ce-cli containerd.io

# Update application dependencies
echo "Updating application dependencies..."
docker-compose pull
pip install -r requirements.txt --upgrade

# Run tests
echo "Running system tests..."
./scripts/run-tests.sh

# Restart services if needed
if [ "$RESTART_REQUIRED" = "true" ]; then
    echo "Restarting services..."
    docker-compose restart
fi

echo "System update completed!"
```

### Performance Optimization

```bash
#!/bin/bash
# optimize-performance.sh

echo "Running performance optimization..."

# Clear old logs
find /var/log -type f -name "*.log" -mtime +7 -delete

# Optimize database
docker-compose exec postgres psql -U postgres -c "VACUUM ANALYZE;"

# Clear Redis cache
docker-compose exec redis redis-cli FLUSHDB

# Clean Docker resources
docker system prune -af --volumes
docker image prune -af

# Optimize disk
fstrim -av

echo "Performance optimization completed!"
```

---

*Document Status: Version 1.0 - January 2025*
*Owner: Platform Operations Lead*
*Review Cycle: Monthly*