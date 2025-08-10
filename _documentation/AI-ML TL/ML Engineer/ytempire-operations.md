# YTEMPIRE Operations

## 6.1 Deployment Procedures

### Deployment Strategy

#### Environment Structure
```yaml
Environments:
  Development:
    Purpose: Active development and testing
    URL: dev.ytempire.local
    Data: Test data only
    Access: Development team only
    Infrastructure: Single server, Docker Compose
    
  Staging:
    Purpose: Pre-production testing
    URL: staging.ytempire.local
    Data: Anonymized production copy
    Access: Extended team + QA
    Infrastructure: Mirrors production setup
    
  Production:
    Purpose: Live system for users
    URL: app.ytempire.com
    Data: Real user data
    Access: Restricted, audit logged
    Infrastructure: High availability setup
```

### Deployment Pipeline

#### CI/CD Configuration
```yaml
# .github/workflows/deploy.yml
name: YTEMPIRE Deployment Pipeline

on:
  push:
    branches: [main, develop]
  pull_request:
    branches: [main]

env:
  REGISTRY: ghcr.io
  IMAGE_NAME: ${{ github.repository }}

jobs:
  test:
    runs-on: ubuntu-latest
    services:
      postgres:
        image: postgres:15
        env:
          POSTGRES_PASSWORD: postgres
        options: >-
          --health-cmd pg_isready
          --health-interval 10s
          --health-timeout 5s
          --health-retries 5
          
      redis:
        image: redis:7
        options: >-
          --health-cmd "redis-cli ping"
          --health-interval 10s
          --health-timeout 5s
          --health-retries 5
          
    steps:
    - uses: actions/checkout@v3
    
    - name: Set up Python
      uses: actions/setup-python@v4
      with:
        python-version: '3.11'
        
    - name: Cache dependencies
      uses: actions/cache@v3
      with:
        path: ~/.cache/pip
        key: ${{ runner.os }}-pip-${{ hashFiles('**/requirements.txt') }}
        
    - name: Install dependencies
      run: |
        python -m pip install --upgrade pip
        pip install -r requirements.txt
        pip install -r requirements-dev.txt
        
    - name: Run linting
      run: |
        flake8 .
        black --check .
        isort --check-only .
        
    - name: Run type checking
      run: mypy .
      
    - name: Run tests
      env:
        DATABASE_URL: postgresql://postgres:postgres@localhost/ytempire_test
        REDIS_URL: redis://localhost:6379
      run: |
        pytest --cov=. --cov-report=xml --cov-report=html
        
    - name: Upload coverage
      uses: codecov/codecov-action@v3
      with:
        file: ./coverage.xml

  build:
    needs: test
    runs-on: ubuntu-latest
    if: github.event_name == 'push'
    
    steps:
    - uses: actions/checkout@v3
    
    - name: Set up Docker Buildx
      uses: docker/setup-buildx-action@v2
      
    - name: Log in to Container Registry
      uses: docker/login-action@v2
      with:
        registry: ${{ env.REGISTRY }}
        username: ${{ github.actor }}
        password: ${{ secrets.GITHUB_TOKEN }}
        
    - name: Extract metadata
      id: meta
      uses: docker/metadata-action@v4
      with:
        images: ${{ env.REGISTRY }}/${{ env.IMAGE_NAME }}
        
    - name: Build and push Docker image
      uses: docker/build-push-action@v4
      with:
        context: .
        platforms: linux/amd64
        push: true
        tags: ${{ steps.meta.outputs.tags }}
        labels: ${{ steps.meta.outputs.labels }}
        cache-from: type=gha
        cache-to: type=gha,mode=max

  deploy:
    needs: build
    runs-on: ubuntu-latest
    if: github.ref == 'refs/heads/main'
    
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
          docker-compose up -d --remove-orphans
          docker-compose exec -T app python manage.py migrate
          docker system prune -f
```

### Deployment Process

#### Blue-Green Deployment
```bash
#!/bin/bash
# Blue-Green Deployment Script

set -e

BLUE_PORT=8000
GREEN_PORT=8001
CURRENT_COLOR=$(cat /opt/ytempire/current_color.txt)

if [ "$CURRENT_COLOR" == "blue" ]; then
    NEW_COLOR="green"
    NEW_PORT=$GREEN_PORT
    OLD_PORT=$BLUE_PORT
else
    NEW_COLOR="blue"
    NEW_PORT=$BLUE_PORT
    OLD_PORT=$GREEN_PORT
fi

echo "Deploying to $NEW_COLOR environment..."

# Deploy new version
docker-compose -f docker-compose.$NEW_COLOR.yml up -d

# Health check
for i in {1..30}; do
    if curl -f http://localhost:$NEW_PORT/health; then
        echo "Health check passed"
        break
    fi
    echo "Waiting for service to be healthy..."
    sleep 10
done

# Switch traffic
echo "Switching traffic to $NEW_COLOR..."
sed -i "s/proxy_pass http:\/\/localhost:$OLD_PORT/proxy_pass http:\/\/localhost:$NEW_PORT/" /etc/nginx/sites-available/ytempire
nginx -s reload

# Update current color
echo $NEW_COLOR > /opt/ytempire/current_color.txt

# Stop old environment after 5 minutes
sleep 300
docker-compose -f docker-compose.$CURRENT_COLOR.yml down

echo "Deployment completed successfully"
```

#### Rollback Procedure
```bash
#!/bin/bash
# Rollback Script

set -e

PREVIOUS_VERSION=$(git rev-parse HEAD~1)
CURRENT_VERSION=$(git rev-parse HEAD)

echo "Rolling back from $CURRENT_VERSION to $PREVIOUS_VERSION"

# Backup current state
docker-compose exec db pg_dump ytempire > /backups/rollback_$(date +%Y%m%d_%H%M%S).sql

# Checkout previous version
git checkout $PREVIOUS_VERSION

# Rebuild and deploy
docker-compose build
docker-compose up -d

# Verify rollback
if docker-compose exec app python manage.py check; then
    echo "Rollback successful"
else
    echo "Rollback failed - manual intervention required"
    exit 1
fi

# Notify team
curl -X POST $SLACK_WEBHOOK_URL \
    -H 'Content-Type: application/json' \
    -d "{\"text\":\"⚠️ Rollback executed from $CURRENT_VERSION to $PREVIOUS_VERSION\"}"
```

### Database Migrations

#### Migration Strategy
```python
# Migration Management
class MigrationManager:
    def __init__(self):
        self.connection = psycopg2.connect(os.environ['DATABASE_URL'])
        
    def run_migrations(self):
        """Execute pending migrations"""
        try:
            # Create migrations table if not exists
            self.create_migrations_table()
            
            # Get pending migrations
            pending = self.get_pending_migrations()
            
            for migration in pending:
                self.execute_migration(migration)
                self.record_migration(migration)
                
            print(f"Successfully applied {len(pending)} migrations")
            
        except Exception as e:
            print(f"Migration failed: {e}")
            self.rollback()
            raise
            
    def rollback_migration(self, version):
        """Rollback specific migration"""
        migration = self.get_migration(version)
        
        if migration.get('down'):
            self.connection.execute(migration['down'])
            self.remove_migration_record(version)
            print(f"Rolled back migration {version}")
        else:
            print(f"No rollback available for {version}")
```

#### Zero-Downtime Migrations
```sql
-- Example: Adding column without downtime
-- Step 1: Add nullable column
ALTER TABLE videos ADD COLUMN new_field VARCHAR(255);

-- Step 2: Backfill data in batches
UPDATE videos SET new_field = 'default_value' 
WHERE new_field IS NULL 
LIMIT 1000;

-- Step 3: Add constraint after backfill
ALTER TABLE videos ALTER COLUMN new_field SET NOT NULL;
```

### Deployment Checklist

```yaml
Pre-Deployment Checklist:
  Code Review:
    ✓ All PRs approved
    ✓ No critical issues in static analysis
    ✓ Security scan passed
    
  Testing:
    ✓ All unit tests passing
    ✓ Integration tests passing
    ✓ Performance benchmarks met
    ✓ Load test completed
    
  Documentation:
    ✓ API documentation updated
    ✓ Changelog updated
    ✓ Release notes prepared
    
  Infrastructure:
    ✓ Database backup completed
    ✓ Resource capacity verified
    ✓ Monitoring alerts configured
    
Post-Deployment Checklist:
  Verification:
    ✓ Health checks passing
    ✓ Key features tested
    ✓ Performance metrics normal
    ✓ No critical errors in logs
    
  Communication:
    ✓ Team notified
    ✓ Status page updated
    ✓ Customer communication sent (if needed)
```

## 6.2 Monitoring & Alerting

### Monitoring Stack

#### Prometheus Configuration
```yaml
# prometheus.yml
global:
  scrape_interval: 30s
  evaluation_interval: 30s
  external_labels:
    monitor: 'ytempire-prod'

scrape_configs:
  - job_name: 'ytempire-api'
    static_configs:
      - targets: ['ytempire-api:8000']
    metrics_path: /metrics
    
  - job_name: 'node-exporter'
    static_configs:
      - targets: ['node-exporter:9100']
      
  - job_name: 'postgres-exporter'
    static_configs:
      - targets: ['postgres-exporter:9187']
      
  - job_name: 'redis-exporter'
    static_configs:
      - targets: ['redis-exporter:9121']
      
  - job_name: 'nvidia-gpu-exporter'
    static_configs:
      - targets: ['nvidia-gpu-exporter:9835']

alerting:
  alertmanagers:
    - static_configs:
        - targets: ['alertmanager:9093']

rule_files:
  - '/etc/prometheus/alerts/*.yml'
```

#### Alert Rules
```yaml
# alerts.yml
groups:
  - name: critical
    interval: 30s
    rules:
      - alert: ServiceDown
        expr: up == 0
        for: 1m
        labels:
          severity: critical
        annotations:
          summary: "Service {{ $labels.job }} is down"
          description: "{{ $labels.instance }} has been down for more than 1 minute"
          
      - alert: HighErrorRate
        expr: rate(http_requests_total{status=~"5.."}[5m]) > 0.05
        for: 5m
        labels:
          severity: critical
        annotations:
          summary: "High error rate detected"
          description: "Error rate is {{ $value }} (>5%) for 5 minutes"
          
      - alert: DatabaseDown
        expr: pg_up == 0
        for: 1m
        labels:
          severity: critical
        annotations:
          summary: "PostgreSQL is down"
          description: "Database has been unreachable for 1 minute"
          
      - alert: VideoGenerationStuck
        expr: video_generation_duration_seconds > 600
        for: 5m
        labels:
          severity: critical
        annotations:
          summary: "Video generation stuck"
          description: "Video generation taking >10 minutes"
          
  - name: warning
    interval: 1m
    rules:
      - alert: HighCPUUsage
        expr: rate(process_cpu_seconds_total[5m]) > 0.8
        for: 10m
        labels:
          severity: warning
        annotations:
          summary: "High CPU usage"
          description: "CPU usage is above 80% for 10 minutes"
          
      - alert: HighMemoryUsage
        expr: (1 - (node_memory_MemAvailable_bytes / node_memory_MemTotal_bytes)) > 0.9
        for: 5m
        labels:
          severity: warning
        annotations:
          summary: "High memory usage"
          description: "Memory usage is above 90%"
          
      - alert: DiskSpaceLow
        expr: (node_filesystem_avail_bytes{mountpoint="/"} / node_filesystem_size_bytes{mountpoint="/"}) < 0.1
        for: 5m
        labels:
          severity: warning
        annotations:
          summary: "Low disk space"
          description: "Less than 10% disk space remaining"
          
      - alert: HighCostPerVideo
        expr: video_generation_cost > 2.5
        for: 10m
        labels:
          severity: warning
        annotations:
          summary: "High cost per video"
          description: "Video generation cost exceeding $2.50"
          
      - alert: YouTubeQuotaWarning
        expr: youtube_quota_usage / youtube_quota_limit > 0.8
        for: 5m
        labels:
          severity: warning
        annotations:
          summary: "YouTube quota usage high"
          description: "YouTube API quota above 80%"
```

### Grafana Dashboards

#### System Overview Dashboard
```json
{
  "dashboard": {
    "title": "YTEMPIRE System Overview",
    "panels": [
      {
        "title": "API Request Rate",
        "targets": [
          {
            "expr": "rate(http_requests_total[5m])",
            "legendFormat": "{{method}} {{endpoint}}"
          }
        ],
        "gridPos": { "h": 8, "w": 12, "x": 0, "y": 0 }
      },
      {
        "title": "Response Time (p95)",
        "targets": [
          {
            "expr": "histogram_quantile(0.95, rate(http_request_duration_seconds_bucket[5m]))",
            "legendFormat": "p95 latency"
          }
        ],
        "gridPos": { "h": 8, "w": 12, "x": 12, "y": 0 }
      },
      {
        "title": "Video Generation Rate",
        "targets": [
          {
            "expr": "rate(videos_generated_total[1h])",
            "legendFormat": "Videos/hour"
          }
        ],
        "gridPos": { "h": 8, "w": 12, "x": 0, "y": 8 }
      },
      {
        "title": "Cost per Video",
        "targets": [
          {
            "expr": "avg(video_generation_cost)",
            "legendFormat": "Average Cost"
          }
        ],
        "gridPos": { "h": 8, "w": 12, "x": 12, "y": 8 }
      },
      {
        "title": "GPU Utilization",
        "targets": [
          {
            "expr": "nvidia_gpu_utilization",
            "legendFormat": "GPU {{gpu}}"
          }
        ],
        "gridPos": { "h": 8, "w": 12, "x": 0, "y": 16 }
      },
      {
        "title": "Active Channels",
        "targets": [
          {
            "expr": "active_channels_total",
            "legendFormat": "Total Channels"
          }
        ],
        "gridPos": { "h": 8, "w": 12, "x": 12, "y": 16 }
      }
    ]
  }
}
```

### Application Monitoring

#### Custom Metrics
```python
from prometheus_client import Counter, Histogram, Gauge, Summary

# Define metrics
request_count = Counter(
    'http_requests_total',
    'Total HTTP requests',
    ['method', 'endpoint', 'status']
)

request_duration = Histogram(
    'http_request_duration_seconds',
    'HTTP request duration',
    ['method', 'endpoint']
)

active_channels = Gauge(
    'active_channels_total',
    'Number of active channels'
)

video_generation_time = Summary(
    'video_generation_seconds',
    'Time to generate video'
)

video_generation_cost = Gauge(
    'video_generation_cost',
    'Cost per video in USD'
)

youtube_quota_usage = Gauge(
    'youtube_quota_usage',
    'YouTube API quota usage'
)

youtube_quota_limit = Gauge(
    'youtube_quota_limit',
    'YouTube API quota limit'
)

# Use in application
@app.route('/api/videos/generate')
@track_metrics
async def generate_video():
    with video_generation_time.time():
        result = await video_service.generate()
        video_generation_cost.set(result['cost'])
    return result
```

### Log Management

#### Structured Logging
```python
import structlog

# Configure structured logging
structlog.configure(
    processors=[
        structlog.stdlib.filter_by_level,
        structlog.stdlib.add_logger_name,
        structlog.stdlib.add_log_level,
        structlog.stdlib.PositionalArgumentsFormatter(),
        structlog.processors.TimeStamper(fmt="iso"),
        structlog.processors.StackInfoRenderer(),
        structlog.processors.format_exc_info,
        structlog.processors.UnicodeDecoder(),
        structlog.processors.JSONRenderer()
    ],
    context_class=dict,
    logger_factory=structlog.stdlib.LoggerFactory(),
    cache_logger_on_first_use=True,
)

logger = structlog.get_logger()

# Use structured logging
logger.info(
    "video_generated",
    channel_id=channel_id,
    video_id=video_id,
    duration=duration,
    cost=cost,
    quality_score=quality_score
)
```

#### Log Aggregation
```yaml
# fluentd.conf
<source>
  @type forward
  port 24224
  bind 0.0.0.0
</source>

<filter docker.**>
  @type parser
  key_name log
  format json
</filter>

<match docker.**>
  @type elasticsearch
  host elasticsearch
  port 9200
  logstash_format true
  logstash_prefix docker
  include_tag_key true
  type_name docker
  tag_key @log_name
  flush_interval 1s
</match>
```

### Alert Management

#### Alertmanager Configuration
```yaml
# alertmanager.yml
global:
  resolve_timeout: 5m
  slack_api_url: '${SLACK_WEBHOOK_URL}'

route:
  group_by: ['alertname', 'cluster', 'service']
  group_wait: 10s
  group_interval: 10s
  repeat_interval: 12h
  receiver: 'default'
  
  routes:
  - match:
      severity: critical
    receiver: 'pagerduty'
    
  - match:
      severity: warning
    receiver: 'slack'

receivers:
- name: 'default'
  webhook_configs:
  - url: 'http://localhost:5001/'
    
- name: 'pagerduty'
  pagerduty_configs:
  - service_key: '${PAGERDUTY_SERVICE_KEY}'
    
- name: 'slack'
  slack_configs:
  - channel: '#alerts'
    title: 'YTEMPIRE Alert'
    text: '{{ range .Alerts }}{{ .Annotations.summary }}{{ end }}'
```

## 6.3 Disaster Recovery

### Backup Strategy

#### Backup Configuration
```yaml
Backup Schedule:
  Database:
    Type: PostgreSQL
    Frequency:
      - Incremental: Every hour
      - Full: Daily at 2 AM
      - Transaction logs: Continuous (WAL archiving)
    Retention:
      - Local: 7 days
      - Remote: 30 days
      - Archive: 1 year
    Encryption: AES-256
    
  Files:
    Type: Generated content
    Frequency:
      - Videos: After generation
      - Thumbnails: After generation
      - User uploads: Real-time sync
    Retention:
      - Local: 7 days
      - Cloud: 90 days
    
  Configuration:
    Type: System configuration
    Frequency:
      - On change (Git)
      - Daily snapshot
    Retention:
      - Git history: Forever
      - Snapshots: 30 days
      
  Redis:
    Type: Cache and queue data
    Frequency:
      - Snapshots: Every 6 hours
      - AOF: Continuous
    Retention:
      - Local: 3 days
```

#### Backup Implementation
```bash
#!/bin/bash
# backup.sh - Automated backup script

set -e

BACKUP_DIR="/backups"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
S3_BUCKET="s3://ytempire-backups"

# Database backup
echo "Starting database backup..."
pg_dump $DATABASE_URL | gzip > $BACKUP_DIR/db_$TIMESTAMP.sql.gz

# Verify backup
if gunzip -c $BACKUP_DIR/db_$TIMESTAMP.sql.gz | head -n 1 | grep -q "PostgreSQL"; then
    echo "Database backup verified"
else
    echo "Database backup verification failed"
    exit 1
fi

# File backup
echo "Starting file backup..."
tar -czf $BACKUP_DIR/files_$TIMESTAMP.tar.gz /opt/ytempire/media

# Redis backup
echo "Starting Redis backup..."
redis-cli --rdb $BACKUP_DIR/redis_$TIMESTAMP.rdb

# Upload to cloud
echo "Uploading to cloud storage..."
aws s3 cp $BACKUP_DIR/db_$TIMESTAMP.sql.gz $S3_BUCKET/database/
aws s3 cp $BACKUP_DIR/files_$TIMESTAMP.tar.gz $S3_BUCKET/files/
aws s3 cp $BACKUP_DIR/redis_$TIMESTAMP.rdb $S3_BUCKET/redis/

# Clean old local backups
find $BACKUP_DIR -name "*.gz" -mtime +7 -delete
find $BACKUP_DIR -name "*.rdb" -mtime +3 -delete

# Send notification
curl -X POST $SLACK_WEBHOOK_URL \
    -H 'Content-Type: application/json' \
    -d '{"text":"✅ Backup completed successfully"}'

echo "Backup completed at $(date)"
```

### Recovery Procedures

#### Recovery Time Objectives
- **RTO (Recovery Time Objective)**: 4 hours
- **RPO (Recovery Point Objective)**: 1 hour
- **MTTR (Mean Time To Recovery)**: 2 hours

#### Recovery Runbook
```yaml
Disaster Recovery Runbook:
  
  1. Assessment (15 minutes):
    - Identify failure scope
    - Determine data loss extent
    - Notify stakeholders
    - Activate incident response team
    
  2. Infrastructure Recovery (1 hour):
    - Provision replacement hardware/VMs
    - Install base operating system
    - Configure network settings
    - Install Docker and dependencies
    
  3. Application Recovery (1 hour):
    - Pull latest application images
    - Deploy docker-compose stack
    - Configure environment variables
    - Verify service connectivity
    
  4. Data Recovery (1.5 hours):
    - Identify latest clean backup
    - Download from cloud storage
    - Restore database
    - Restore file system
    - Restore Redis cache
    - Verify data integrity
    
  5. Validation (30 minutes):
    - Run health checks
    - Verify critical functions
    - Test user workflows
    - Monitor for errors
    
  6. Communication:
    - Update status page
    - Notify users of restoration
    - Document incident
    - Schedule post-mortem
```

#### Automated Recovery Script
```python
class DisasterRecovery:
    def __init__(self):
        self.backup_manager = BackupManager()
        self.health_checker = HealthChecker()
        self.notification_service = NotificationService()
        
    async def execute_recovery(self, failure_type):
        """Execute disaster recovery procedure"""
        
        recovery_plan = self.get_recovery_plan(failure_type)
        
        try:
            # Phase 1: Stop affected services
            await self.stop_services(recovery_plan['affected_services'])
            
            # Phase 2: Restore from backup
            if recovery_plan.get('restore_database'):
                await self.restore_database()
                
            if recovery_plan.get('restore_files'):
                await self.restore_files()
                
            if recovery_plan.get('restore_cache'):
                await self.restore_redis()
                
            # Phase 3: Restart services
            await self.start_services(recovery_plan['affected_services'])
            
            # Phase 4: Validate recovery
            health_status = await self.health_checker.check_all()
            
            if health_status['healthy']:
                await self.notification_service.send(
                    "Recovery completed successfully",
                    severity='info'
                )
                return True
            else:
                await self.notification_service.send(
                    f"Recovery completed with issues: {health_status['issues']}",
                    severity='warning'
                )
                return False
                
        except Exception as e:
            await self.notification_service.send(
                f"Recovery failed: {str(e)}",
                severity='critical'
            )
            raise
            
    async def restore_database(self):
        """Restore PostgreSQL database"""
        latest_backup = await self.backup_manager.get_latest_backup('database')
        
        # Download backup
        local_file = await self.backup_manager.download(latest_backup)
        
        # Stop database
        subprocess.run(['docker-compose', 'stop', 'postgres'])
        
        # Restore
        subprocess.run([
            'gunzip', '-c', local_file, '|',
            'docker-compose', 'exec', '-T', 'postgres',
            'psql', '-U', 'postgres', 'ytempire'
        ], shell=True)
        
        # Start database
        subprocess.run(['docker-compose', 'start', 'postgres'])
        
        # Verify
        await self.verify_database_integrity()
```

### Business Continuity Plan

#### Incident Response Team
```yaml
Incident Response Team:
  Incident Commander:
    Role: Overall coordination
    Responsibilities:
      - Declare incident
      - Coordinate response
      - External communication
      - Decision authority
    
  Technical Lead:
    Role: Technical response
    Responsibilities:
      - Assess technical impact
      - Direct recovery efforts
      - Validate restoration
    
  Operations Lead:
    Role: Infrastructure
    Responsibilities:
      - Server/network recovery
      - Backup restoration
      - System monitoring
    
  Communications Lead:
    Role: Stakeholder updates
    Responsibilities:
      - Customer communication
      - Team updates
      - Status page updates
```

#### Communication Plan
```yaml
Communication Matrix:
  Internal:
    Immediate (0-15 min):
      - Incident channel created
      - Core team notified
      - Initial assessment shared
    
    Regular Updates (Every 30 min):
      - Progress reports
      - Blockers identified
      - Next steps defined
    
  External:
    Initial (Within 30 min):
      - Status page updated
      - Twitter/social media post
      - Email to affected users
    
    Regular (Hourly):
      - Status page updates
      - Email updates for major changes
    
    Resolution:
      - Full service restoration notice
      - Post-mortem scheduled
      - Lessons learned shared
```

## 6.4 Scaling Guidelines

### Scaling Triggers

#### Automatic Scaling Rules
```yaml
Scaling Triggers:
  CPU-based:
    Scale Up: CPU > 80% for 5 minutes
    Scale Down: CPU < 30% for 10 minutes
    Cool-down: 5 minutes
    
  Memory-based:
    Scale Up: Memory > 85% for 5 minutes
    Scale Down: Memory < 40% for 10 minutes
    Cool-down: 5 minutes
    
  Queue-based:
    Scale Up: Queue depth > 50 videos
    Scale Down: Queue depth < 10 videos
    Cool-down: 10 minutes
    
  Traffic-based:
    Scale Up: Requests/sec > 100
    Scale Down: Requests/sec < 20
    Cool-down: 5 minutes
    
  Business-based:
    Scale Up: Active channels > 200
    Scale Down: Active channels < 50
    Cool-down: 15 minutes
    
  GPU-based:
    Scale Up: GPU utilization > 90% for 10 minutes
    Scale Down: GPU utilization < 30% for 20 minutes
    Cool-down: 15 minutes
```

#### Scaling Implementation
```python
class AutoScaler:
    def __init__(self):
        self.min_instances = 1
        self.max_instances = 10
        self.current_instances = 1
        self.metrics_client = MetricsClient()
        self.cooldown_periods = {}
        
    async def evaluate_scaling(self):
        """Evaluate if scaling is needed"""
        
        metrics = await self.metrics_client.get_current_metrics()
        
        # Check cooldown
        if self.in_cooldown():
            return
            
        # Check scaling triggers
        if self.should_scale_up(metrics):
            await self.scale_up()
        elif self.should_scale_down(metrics):
            await self.scale_down()
            
    def should_scale_up(self, metrics):
        """Determine if scale up is needed"""
        triggers = [
            metrics['cpu'] > 80,
            metrics['memory'] > 85,
            metrics['queue_depth'] > 50,
            metrics['rps'] > 100,
            metrics['gpu_util'] > 90
        ]
        return any(triggers) and self.current_instances < self.max_instances
        
    async def scale_up(self):
        """Add more instances"""
        
        new_instance_count = min(
            self.current_instances + 1,
            self.max_instances
        )
        
        # Launch new container
        subprocess.run([
            'docker-compose',
            'up',
            '-d',
            '--scale',
            f'worker={new_instance_count}'
        ])
        
        self.current_instances = new_instance_count
        self.set_cooldown()
        
        logger.info(f"Scaled up to {self.current_instances} instances")
        
    async def scale_down(self):
        """Remove instances"""
        
        if self.current_instances <= self.min_instances:
            return
            
        new_instance_count = self.current_instances - 1
        
        # Gracefully stop container
        subprocess.run([
            'docker-compose',
            'up',
            '-d',
            '--scale',
            f'worker={new_instance_count}'
        ])
        
        self.current_instances = new_instance_count
        self.set_cooldown()
        
        logger.info(f"Scaled down to {self.current_instances} instances")
```

### Capacity Planning

#### Growth Projections
```yaml
Capacity Planning:
  Current (MVP):
    Users: 50
    Channels: 250
    Videos/Day: 50
    Storage: 6TB
    Bandwidth: 100GB/day
    Cost/Month: $5,000
    
  3 Months:
    Users: 200
    Channels: 1,000
    Videos/Day: 150
    Storage: 20TB
    Bandwidth: 500GB/day
    Cost/Month: $15,000
    
  6 Months:
    Users: 1,000
    Channels: 5,000
    Videos/Day: 300
    Storage: 50TB
    Bandwidth: 1TB/day
    Cost/Month: $30,000
    
  12 Months:
    Users: 5,000
    Channels: 25,000
    Videos/Day: 1,000
    Storage: 200TB
    Bandwidth: 5TB/day
    Cost/Month: $100,000
```

#### Resource Planning
```python
def calculate_resources(users, channels, videos_per_day):
    """Calculate required resources based on load"""
    
    resources = {
        'cpu_cores': math.ceil(users / 10),
        'memory_gb': math.ceil(channels / 2),
        'gpu_hours': videos_per_day * 0.1,  # 6 minutes per video
        'storage_tb': (videos_per_day * 0.5 * 30) / 1000,  # 500MB per video, 30 days
        'bandwidth_gb': videos_per_day * 2,  # 2GB per video
        'database_connections': min(users * 2, 500),
        'redis_memory_gb': math.ceil(users / 50),
        'worker_instances': math.ceil(videos_per_day / 50),
        'api_instances': math.ceil(users / 100)
    }
    
    # Cost estimation
    costs = {
        'compute': resources['cpu_cores'] * 50 + resources['memory_gb'] * 10,
        'gpu': resources['gpu_hours'] * 0.50,
        'storage': resources['storage_tb'] * 100,
        'bandwidth': resources['bandwidth_gb'] * 0.02,
        'total_monthly': 0
    }
    costs['total_monthly'] = sum(costs.values()) * 30
    
    return resources, costs
```

### Migration to Cloud

#### Cloud Migration Plan
```yaml
Cloud Migration Phases:
  
  Phase 1 - Hybrid (Month 4-5):
    Infrastructure:
      - Keep core services local
      - Move backups to cloud
      - Use cloud for burst capacity
      - Test cloud providers
    
    Actions:
      - Set up AWS/GCP accounts
      - Configure S3/GCS for backups
      - Test cloud GPU instances
      - Implement hybrid networking
    
    Success Criteria:
      - Backups successfully in cloud
      - Can burst to cloud for processing
      - Network latency acceptable
    
  Phase 2 - Database (Month 6):
    Infrastructure:
      - Migrate to managed PostgreSQL
      - Set up read replicas
      - Implement connection pooling
      - Test failover procedures
    
    Actions:
      - Set up AWS RDS/Cloud SQL
      - Migrate data with minimal downtime
      - Configure automated backups
      - Test disaster recovery
    
    Success Criteria:
      - Zero data loss
      - <5 minute switchover
      - Performance maintained
    
  Phase 3 - Application (Month 7-8):
    Infrastructure:
      - Containerize all services
      - Deploy to Kubernetes
      - Implement auto-scaling
      - Set up load balancing
    
    Actions:
      - Create Helm charts
      - Deploy to EKS/GKE
      - Configure ingress
      - Implement service mesh
    
    Success Criteria:
      - All services running in K8s
      - Auto-scaling functional
      - Zero-downtime deployments
    
  Phase 4 - ML/GPU (Month 9-10):
    Infrastructure:
      - Move ML workloads to cloud
      - Use spot instances for training
      - Implement model caching
      - Optimize for cost
    
    Actions:
      - Set up GPU node pools
      - Migrate ML models
      - Implement spot instance handling
      - Set up model registry
    
    Success Criteria:
      - ML inference <100ms
      - 50% cost reduction on GPU
      - Models versioned and cached
    
  Phase 5 - Full Cloud (Month 11-12):
    Infrastructure:
      - Complete migration
      - Decommission local hardware
      - Multi-region deployment
      - Global CDN implementation
    
    Actions:
      - Final cutover
      - Sell/repurpose hardware
      - Set up multi-region
      - Configure CDN
    
    Success Criteria:
      - 100% cloud-based
      - 99.99% availability
      - Global <100ms latency
```

#### Kubernetes Configuration
```yaml
# deployment.yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: ytempire-api
spec:
  replicas: 3
  selector:
    matchLabels:
      app: ytempire-api
  template:
    metadata:
      labels:
        app: ytempire-api
    spec:
      containers:
      - name: api
        image: ytempire/api:latest
        ports:
        - containerPort: 8000
        env:
        - name: DATABASE_URL
          valueFrom:
            secretKeyRef:
              name: ytempire-secrets
              key: database-url
        resources:
          requests:
            memory: "1Gi"
            cpu: "500m"
          limits:
            memory: "2Gi"
            cpu: "1000m"
        livenessProbe:
          httpGet:
            path: /health
            port: 8000
          initialDelaySeconds: 30
          periodSeconds: 10
        readinessProbe:
          httpGet:
            path: /ready
            port: 8000
          initialDelaySeconds: 5
          periodSeconds: 5
---
apiVersion: v1
kind: Service
metadata:
  name: ytempire-api
spec:
  selector:
    app: ytempire-api
  ports:
  - port: 80
    targetPort: 8000
  type: LoadBalancer
---
apiVersion: autoscaling/v2
kind: HorizontalPodAutoscaler
metadata:
  name: ytempire-api-hpa
spec:
  scaleTargetRef:
    apiVersion: apps/v1
    kind: Deployment
    name: ytempire-api
  minReplicas: 3
  maxReplicas: 10
  metrics:
  - type: Resource
    resource:
      name: cpu
      target:
        type: Utilization
        averageUtilization: 70
  - type: Resource
    resource:
      name: memory
      target:
        type: Utilization
        averageUtilization: 80
```

## 6.5 Cost Optimization

### Cost Tracking

#### Cost Breakdown Structure
```python
COST_STRUCTURE = {
    'infrastructure': {
        'compute': {
            'cpu_hour': 0.05,
            'gpu_hour': 0.50,
            'memory_gb_hour': 0.01
        },
        'storage': {
            'ssd_gb_month': 0.10,
            'hdd_gb_month': 0.02,
            'backup_gb_month': 0.01,
            'bandwidth_gb': 0.02
        },
        'network': {
            'data_transfer_gb': 0.02,
            'cdn_gb': 0.05,
            'load_balancer_hour': 0.025
        }
    },
    'apis': {
        'openai': {
            'gpt_3_5_1k_tokens': 0.002,
            'gpt_4_1k_tokens': 0.06,
            'embedding_1k_tokens': 0.0001
        },
        'elevenlabs': {
            'character': 0.0003,
            'voice_clone': 5.00
        },
        'google_tts': {
            'character': 0.000004
        },
        'stable_diffusion': {
            'image': 0.002
        },
        'youtube': {
            'quota_unit': 0.0001  # Estimated cost
        }
    },
    'operational': {
        'monitoring': 50.00,  # Monthly
        'backups': 100.00,   # Monthly
        'ssl_certificates': 0.00,  # Let's Encrypt
        'domain': 15.00  # Monthly
    }
}
```

#### Cost Tracking Implementation
```python
class CostTracker:
    def __init__(self):
        self.cost_db = CostDatabase()
        self.alert_thresholds = {
            'video': 3.00,  # $3 per video
            'daily': 200.00,  # $200 per day
            'monthly': 5000.00  # $5000 per month
        }
        
    async def track_video_cost(self, video_id, costs):
        """Track costs for video generation"""
        
        breakdown = {
            'script_generation': costs.get('openai', 0),
            'voice_synthesis': costs.get('voice', 0),
            'thumbnail': costs.get('image', 0),
            'processing': costs.get('compute', 0),
            'storage': costs.get('storage', 0)
        }
        
        total_cost = sum(breakdown.values())
        
        # Record in database
        await self.cost_db.insert({
            'video_id': video_id,
            'timestamp': datetime.now(),
            'breakdown': breakdown,
            'total': total_cost
        })
        
        # Check against threshold
        if total_cost > self.alert_thresholds['video']:
            await self.send_cost_alert('video', video_id, total_cost)
            
        # Update running totals
        await self.update_daily_total(total_cost)
        await self.update_monthly_total(total_cost)
        
        return total_cost
        
    async def get_cost_report(self, period='daily'):
        """Generate cost report"""
        
        costs = await self.cost_db.get_costs(period)
        
        report = {
            'period': period,
            'total_cost': sum(c['total'] for c in costs),
            'video_count': len(costs),
            'average_cost': statistics.mean([c['total'] for c in costs]) if costs else 0,
            'breakdown': self.aggregate_breakdown(costs),
            'trend': self.calculate_trend(costs),
            'top_expenses': self.get_top_expenses(costs),
            'optimization_opportunities': self.identify_savings(costs)
        }
        
        return report
        
    def identify_savings(self, costs):
        """Identify cost optimization opportunities"""
        
        opportunities = []
        
        # Check API usage patterns
        api_costs = [c['breakdown'].get('script_generation', 0) for c in costs]
        if api_costs and max(api_costs) > 0.50:
            opportunities.append({
                'area': 'API Usage',
                'potential_savings': sum(api_costs) * 0.3,
                'recommendation': 'Switch to GPT-3.5 for simple scripts'
            })
            
        # Check voice synthesis
        voice_costs = [c['breakdown'].get('voice_synthesis', 0) for c in costs]
        if voice_costs and statistics.mean(voice_costs) > 0.20:
            opportunities.append({
                'area': 'Voice Synthesis',
                'potential_savings': sum(voice_costs) * 0.5,
                'recommendation': 'Use Google TTS for non-critical videos'
            })
            
        return opportunities
```

### Optimization Strategies

#### API Cost Optimization
```python
class APIOptimizer:
    def __init__(self):
        self.cache = RedisCache()
        self.model_selector = ModelSelector()
        
    async def optimize_llm_request(self, prompt, quality_required):
        """Select optimal LLM based on requirements"""
        
        # Check cache first
        cache_key = hashlib.md5(prompt.encode()).hexdigest()
        cached = await self.cache.get(cache_key)
        if cached:
            return cached, 0.00  # No cost for cached response
            
        # Select model based on quality requirements
        model_config = {
            'low': {
                'model': 'gpt-3.5-turbo',
                'cost_per_1k': 0.002,
                'max_tokens': 1000
            },
            'medium': {
                'model': 'gpt-3.5-turbo-16k',
                'cost_per_1k': 0.003,
                'max_tokens': 2000
            },
            'high': {
                'model': 'gpt-4',
                'cost_per_1k': 0.06,
                'max_tokens': 4000
            }
        }
        
        quality_tier = 'low' if quality_required < 0.7 else 'medium' if quality_required < 0.9 else 'high'
        config = model_config[quality_tier]
        
        # Make request
        response = await self.make_request(config['model'], prompt, config['max_tokens'])
        
        # Cache response
        await self.cache.set(cache_key, response, ttl=3600)
        
        # Calculate cost
        tokens = len(prompt.split()) * 1.3  # Estimate
        cost = (tokens / 1000) * config['cost_per_1k']
        
        return response, cost
        
    async def batch_requests(self, requests):
        """Batch multiple requests for efficiency"""
        
        # Group by model type
        grouped = {}
        for req in requests:
            model = self.select_model(req['quality'])
            if model not in grouped:
                grouped[model] = []
            grouped[model].append(req)
            
        # Process batches
        results = []
        for model, batch in grouped.items():
            # Combine prompts with delimiters
            combined_prompt = '\n---SEPARATOR---\n'.join([r['prompt'] for r in batch])
            
            # Single API call
            response = await self.make_request(model, combined_prompt)
            
            # Split responses
            split_responses = response.split('---SEPARATOR---')
            
            for i, resp in enumerate(split_responses):
                results.append({
                    'id': batch[i]['id'],
                    'response': resp,
                    'model': model
                })
                
        return results
```

#### Infrastructure Optimization
```yaml
Cost Optimization Tactics:
  
  Compute:
    - Use spot instances for batch jobs (70% savings)
    - Right-size instances based on actual usage
    - Implement aggressive auto-scaling down
    - Use ARM processors where possible (20% cheaper)
    - Reserved instances for baseline load (30% savings)
    
  Storage:
    - Compress videos before storage (50% reduction)
    - Use tiered storage (hot/warm/cold)
    - Delete temporary files promptly
    - Implement deduplication
    - Archive old content to glacier
    
  Network:
    - Use CDN for static content
    - Compress API responses (gzip)
    - Implement request batching
    - Cache aggressively
    - Use regional endpoints
    
  APIs:
    - Cache API responses (Redis)
    - Batch API requests
    - Use cheaper alternatives when possible
    - Implement fallback strategies
    - Monitor quota usage closely
    
  Database:
    - Use read replicas for analytics
    - Implement connection pooling
    - Archive old data
    - Optimize queries
    - Use appropriate indexes
```

### Budget Management

#### Budget Allocation
```yaml
Monthly Budget Allocation ($5,000):
  Infrastructure: $1,500 (30%)
    - Compute: $800
    - Storage: $400
    - Network: $300
    
  APIs: $2,500 (50%)
    - OpenAI: $1,500
    - ElevenLabs: $500
    - Other APIs: $500
    
  Operational: $500 (10%)
    - Monitoring: $200
    - Backups: $200
    - Misc: $100
    
  Reserve: $500 (10%)
    - Emergency scaling
    - Unexpected costs
    
Progressive Budget Scaling:
  Month 1-3: $5,000/month
  Month 4-6: $15,000/month
  Month 7-9: $30,000/month
  Month 10-12: $50,000/month
```

#### Cost Alerts
```python
class BudgetManager:
    def __init__(self):
        self.monthly_budget = 5000
        self.alert_thresholds = [0.5, 0.8, 0.9, 1.0]
        self.cost_controls = {
            'soft_limit': 0.9,  # Warning only
            'hard_limit': 1.1   # Stop non-critical operations
        }
        
    async def check_budget(self):
        """Check current spending against budget"""
        
        current_spend = await self.get_current_month_spend()
        budget_used = current_spend / self.monthly_budget
        
        # Check alert thresholds
        for threshold in self.alert_thresholds:
            if budget_used >= threshold and not self.alert_sent(threshold):
                await self.send_budget_alert(threshold, current_spend)
                self.mark_alert_sent(threshold)
                
        # Implement cost controls
        if budget_used >= self.cost_controls['soft_limit']:
            await self.implement_soft_controls()
            
        if budget_used >= self.cost_controls['hard_limit']:
            await self.implement_hard_controls()
            
    async def implement_soft_controls(self):
        """Implement soft cost controls"""
        
        # Switch to cheaper alternatives
        await self.switch_to_cheaper_models()
        
        # Reduce quality thresholds
        await self.reduce_quality_thresholds()
        
        # Increase cache TTL
        await self.increase_cache_ttl()
        
        logger.warning("Soft cost controls activated")
        
    async def implement_hard_controls(self):
        """Implement hard cost controls"""
        
        # Stop non-critical operations
        await self.pause_low_priority_channels()
        
        # Disable expensive features
        await self.disable_premium_features()
        
        # Alert administrators
        await self.send_critical_budget_alert()
        
        logger.critical("Hard cost controls activated - budget exceeded")
```

### Cost Optimization Dashboard
```python
class CostDashboard:
    """Real-time cost monitoring dashboard"""
    
    def __init__(self):
        self.metrics = {
            'current_month_spend': 0,
            'current_day_spend': 0,
            'videos_generated_today': 0,
            'average_cost_per_video': 0,
            'api_costs': {},
            'infrastructure_costs': {}
        }
        
    async def get_dashboard_data(self):
        """Get real-time cost dashboard data"""
        
        return {
            'summary': {
                'month_to_date': self.metrics['current_month_spend'],
                'budget_remaining': 5000 - self.metrics['current_month_spend'],
                'budget_used_percent': (self.metrics['current_month_spend'] / 5000) * 100,
                'projected_month_end': self.project_month_end_cost()
            },
            'today': {
                'spend': self.metrics['current_day_spend'],
                'videos': self.metrics['videos_generated_today'],
                'avg_cost': self.metrics['average_cost_per_video']
            },
            'breakdown': {
                'apis': self.metrics['api_costs'],
                'infrastructure': self.metrics['infrastructure_costs']
            },
            'trends': {
                'daily': await self.get_daily_trend(),
                'weekly': await self.get_weekly_trend()
            },
            'optimization': {
                'opportunities': await self.get_optimization_opportunities(),
                'implemented': await self.get_implemented_optimizations()
            }
        }
```