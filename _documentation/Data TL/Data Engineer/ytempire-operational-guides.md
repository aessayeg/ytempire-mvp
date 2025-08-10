# 5. OPERATIONAL GUIDES - YTEMPIRE Documentation

## 5.1 Development Workflow

### Local Development Setup

```bash
#!/bin/bash
# setup-dev.sh - Local development environment setup

# Clone repositories
git clone https://github.com/ytempire/ytempire-platform.git
cd ytempire-platform

# Create Python virtual environment
python3.11 -m venv venv
source venv/bin/activate

# Install Python dependencies
pip install -r requirements.txt
pip install -r requirements-dev.txt

# Install Node.js dependencies
cd frontend
npm install
cd ..

# Copy environment template
cp .env.example .env

# Generate secrets
echo "JWT_SECRET=$(openssl rand -hex 32)" >> .env
echo "POSTGRES_PASSWORD=$(openssl rand -hex 16)" >> .env
echo "REDIS_PASSWORD=$(openssl rand -hex 16)" >> .env

# Start development services
docker-compose -f docker-compose.dev.yml up -d

# Initialize database
python scripts/init_db.py
python scripts/seed_dev_data.py

echo "Development environment ready!"
echo "API: http://localhost:8000"
echo "Frontend: http://localhost:3000"
echo "Grafana: http://localhost:3001"
echo "N8N: http://localhost:5678"
```

### Git Workflow

```yaml
# .github/workflows/ci.yml
name: YTEMPIRE CI/CD Pipeline

on:
  push:
    branches: [main, develop]
  pull_request:
    branches: [main]

jobs:
  test:
    runs-on: ubuntu-latest
    
    services:
      postgres:
        image: timescale/timescaledb:latest-pg15
        env:
          POSTGRES_PASSWORD: test
          POSTGRES_DB: ytempire_test
        options: >-
          --health-cmd pg_isready
          --health-interval 10s
          --health-timeout 5s
          --health-retries 5
        ports:
          - 5432:5432
      
      redis:
        image: redis:7.2-alpine
        options: >-
          --health-cmd "redis-cli ping"
          --health-interval 10s
          --health-timeout 5s
          --health-retries 5
        ports:
          - 6379:6379
    
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
          pip install -r requirements-test.txt
      
      - name: Run linting
        run: |
          flake8 . --count --select=E9,F63,F7,F82 --show-source --statistics
          black --check .
          isort --check-only .
      
      - name: Run type checking
        run: mypy . --ignore-missing-imports
      
      - name: Run tests
        env:
          DATABASE_URL: postgresql://postgres:test@localhost:5432/ytempire_test
          REDIS_URL: redis://localhost:6379/0
        run: |
          pytest tests/ -v --cov=app --cov-report=xml --cov-report=html
      
      - name: Upload coverage
        uses: codecov/codecov-action@v3
        with:
          file: ./coverage.xml
          fail_ci_if_error: true

  build:
    needs: test
    runs-on: ubuntu-latest
    if: github.ref == 'refs/heads/main'
    
    steps:
      - uses: actions/checkout@v3
      
      - name: Set up Docker Buildx
        uses: docker/setup-buildx-action@v2
      
      - name: Build and push Docker images
        run: |
          docker build -t ytempire-api:${{ github.sha }} ./backend
          docker build -t ytempire-frontend:${{ github.sha }} ./frontend
          docker build -t ytempire-ml:${{ github.sha }} ./ml
      
      - name: Save Docker images
        run: |
          docker save ytempire-api:${{ github.sha }} | gzip > api.tar.gz
          docker save ytempire-frontend:${{ github.sha }} | gzip > frontend.tar.gz
          docker save ytempire-ml:${{ github.sha }} | gzip > ml.tar.gz
      
      - name: Upload artifacts
        uses: actions/upload-artifact@v3
        with:
          name: docker-images
          path: |
            api.tar.gz
            frontend.tar.gz
            ml.tar.gz
```

### Code Review Process

```markdown
# Code Review Checklist

## General
- [ ] Code follows project style guidelines
- [ ] Naming conventions are consistent
- [ ] No commented-out code
- [ ] No debug/console.log statements

## Functionality
- [ ] Code works as expected
- [ ] Edge cases are handled
- [ ] Error handling is appropriate
- [ ] No obvious bugs

## Performance
- [ ] No unnecessary database queries
- [ ] Efficient algorithms used
- [ ] Caching implemented where appropriate
- [ ] No memory leaks

## Security
- [ ] Input validation implemented
- [ ] SQL injection prevention
- [ ] XSS prevention
- [ ] Sensitive data not exposed

## Testing
- [ ] Unit tests added/updated
- [ ] Integration tests added/updated
- [ ] Test coverage >= 80%
- [ ] All tests passing

## Documentation
- [ ] Functions/methods documented
- [ ] Complex logic explained
- [ ] README updated if needed
- [ ] API documentation updated

## Database
- [ ] Migrations are reversible
- [ ] Indexes added for queries
- [ ] No N+1 query problems
- [ ] Foreign keys properly defined
```

## 5.2 Testing Strategy

### Unit Testing

```python
# tests/test_content_generation.py
import pytest
from unittest.mock import Mock, patch
from app.services.content_generation import ContentGenerator

class TestContentGeneration:
    """
    Unit tests for content generation service
    """
    
    @pytest.fixture
    def content_generator(self):
        """Create content generator instance"""
        return ContentGenerator()
    
    @pytest.fixture
    def mock_trend_data(self):
        """Mock trend data"""
        return {
            'topic': 'Test Topic',
            'keywords': ['keyword1', 'keyword2'],
            'duration': 8,
            'style': 'educational'
        }
    
    @patch('app.services.content_generation.openai.Client')
    def test_generate_script_success(self, mock_openai, content_generator, mock_trend_data):
        """Test successful script generation"""
        # Setup mock
        mock_response = Mock()
        mock_response.choices[0].message.content = "Generated script content"
        mock_openai.return_value.chat.completions.create.return_value = mock_response
        
        # Call method
        result = content_generator.generate_script(mock_trend_data)
        
        # Assertions
        assert result == "Generated script content"
        mock_openai.return_value.chat.completions.create.assert_called_once()
    
    def test_validate_script_valid(self, content_generator):
        """Test script validation with valid script"""
        valid_script = """
        Hook: Did you know this amazing fact?
        Introduction: Welcome to our channel.
        Main content: Here's the important information.
        Conclusion: That's all for today.
        Call to action: Please subscribe!
        """
        
        assert content_generator.validate_script(valid_script) == True
    
    def test_validate_script_too_short(self, content_generator):
        """Test script validation with too short script"""
        short_script = "This is too short"
        assert content_generator.validate_script(short_script) == False
    
    @pytest.mark.parametrize("quality_score,expected", [
        (90, True),
        (85, True),
        (84, False),
        (50, False)
    ])
    def test_quality_check(self, content_generator, quality_score, expected):
        """Test quality score checking"""
        result = content_generator.passes_quality_check(quality_score)
        assert result == expected
```

### Integration Testing

```python
# tests/test_integration.py
import pytest
from fastapi.testclient import TestClient
from app.main import app

class TestAPIIntegration:
    """
    Integration tests for API endpoints
    """
    
    @pytest.fixture
    def client(self):
        """Create test client"""
        return TestClient(app)
    
    @pytest.fixture
    def auth_headers(self, client):
        """Get authentication headers"""
        response = client.post("/auth/login", json={
            "username": "test_user",
            "password": "test_password"
        })
        token = response.json()["access_token"]
        return {"Authorization": f"Bearer {token}"}
    
    def test_video_generation_workflow(self, client, auth_headers):
        """Test complete video generation workflow"""
        # 1. Create generation request
        response = client.post(
            "/api/videos/generate",
            headers=auth_headers,
            json={
                "channel_id": "test_channel",
                "topic": "Test Topic",
                "duration": 8
            }
        )
        assert response.status_code == 202
        job_id = response.json()["job_id"]
        
        # 2. Check job status
        response = client.get(
            f"/api/jobs/{job_id}",
            headers=auth_headers
        )
        assert response.status_code == 200
        assert response.json()["status"] in ["queued", "processing", "completed"]
        
        # 3. Get video details
        response = client.get(
            f"/api/videos/{job_id}",
            headers=auth_headers
        )
        assert response.status_code in [200, 404]  # 404 if still processing
```

### Load Testing

```javascript
// k6-load-test.js
import http from 'k6/http';
import { check, sleep } from 'k6';

export let options = {
    stages: [
        { duration: '2m', target: 10 },  // Ramp up to 10 users
        { duration: '5m', target: 50 },  // Stay at 50 users
        { duration: '2m', target: 100 }, // Ramp up to 100 users
        { duration: '5m', target: 100 }, // Stay at 100 users
        { duration: '2m', target: 0 },   // Ramp down to 0
    ],
    thresholds: {
        http_req_duration: ['p(95)<2000'], // 95% of requests must complete below 2s
        http_req_failed: ['rate<0.1'],     // Error rate must be below 10%
    },
};

const BASE_URL = 'http://localhost:8000';

export default function () {
    // Test API health endpoint
    let healthCheck = http.get(`${BASE_URL}/health`);
    check(healthCheck, {
        'health check status is 200': (r) => r.status === 200,
    });
    
    // Test video generation endpoint
    let payload = JSON.stringify({
        channel_id: 'test_channel',
        topic: 'Load Test Topic',
        duration: 8,
    });
    
    let params = {
        headers: { 'Content-Type': 'application/json' },
    };
    
    let response = http.post(`${BASE_URL}/api/videos/generate`, payload, params);
    check(response, {
        'video generation status is 202': (r) => r.status === 202,
        'response has job_id': (r) => JSON.parse(r.body).job_id !== undefined,
    });
    
    sleep(1);
}
```

## 5.3 Deployment Process

### Production Deployment Checklist

```markdown
# Production Deployment Checklist

## Pre-Deployment
- [ ] All tests passing in CI/CD
- [ ] Code reviewed and approved
- [ ] Database migrations tested
- [ ] Rollback plan documented
- [ ] Stakeholders notified

## Deployment Steps
1. [ ] Create backup of production database
2. [ ] Tag release in Git
3. [ ] Build Docker images with new tag
4. [ ] Deploy to staging environment
5. [ ] Run smoke tests on staging
6. [ ] Deploy to production (blue-green)
7. [ ] Run health checks
8. [ ] Monitor for 30 minutes

## Post-Deployment
- [ ] Update documentation
- [ ] Close related tickets
- [ ] Send deployment notification
- [ ] Update changelog
- [ ] Clean up old images

## Rollback Criteria
- Error rate > 5%
- Response time > 2s (p95)
- Critical functionality broken
- Database errors
```

### Deployment Script

```bash
#!/bin/bash
# deploy.sh - Production deployment script

set -e

VERSION=$1
ENVIRONMENT=$2

if [ -z "$VERSION" ] || [ -z "$ENVIRONMENT" ]; then
    echo "Usage: ./deploy.sh <version> <environment>"
    exit 1
fi

echo "Deploying version $VERSION to $ENVIRONMENT"

# Pre-deployment checks
./scripts/pre-deploy-checks.sh

# Create backup
./scripts/backup.sh

# Deploy
case $ENVIRONMENT in
    staging)
        echo "Deploying to staging..."
        docker-compose -f docker-compose.staging.yml pull
        docker-compose -f docker-compose.staging.yml up -d
        ;;
    production)
        echo "Deploying to production..."
        ./scripts/blue-green-deploy.py $VERSION
        ;;
    *)
        echo "Unknown environment: $ENVIRONMENT"
        exit 1
        ;;
esac

# Run health checks
./scripts/health-check.sh $ENVIRONMENT

# Run smoke tests
./scripts/smoke-tests.sh $ENVIRONMENT

echo "Deployment complete!"
```

## 5.4 Monitoring & Maintenance

### Daily Monitoring Tasks

```python
#!/usr/bin/env python3
"""
daily_monitoring.py - Daily monitoring and maintenance tasks
"""

import requests
import psycopg2
import redis
import logging
from datetime import datetime, timedelta

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class DailyMonitoring:
    """
    Execute daily monitoring tasks
    """
    
    def __init__(self):
        self.db_conn = psycopg2.connect(
            host="localhost",
            database="ytempire",
            user="ytempire_user",
            password=os.environ["POSTGRES_PASSWORD"]
        )
        self.redis_client = redis.Redis(host="localhost", port=6379)
        
    def check_system_health(self):
        """Check overall system health"""
        checks = {
            'api': self.check_api_health(),
            'database': self.check_database_health(),
            'redis': self.check_redis_health(),
            'disk_space': self.check_disk_space(),
            'video_generation': self.check_video_generation_rate(),
            'cost_tracking': self.check_cost_per_video(),
            'error_rate': self.check_error_rate()
        }
        
        return checks
    
    def check_api_health(self):
        """Check API health"""
        try:
            response = requests.get("http://localhost:8000/health", timeout=5)
            return response.status_code == 200
        except:
            return False
    
    def check_database_health(self):
        """Check database health"""
        try:
            cursor = self.db_conn.cursor()
            cursor.execute("SELECT 1")
            cursor.close()
            return True
        except:
            return False
    
    def check_video_generation_rate(self):
        """Check video generation rate"""
        cursor = self.db_conn.cursor()
        cursor.execute("""
            SELECT COUNT(*) 
            FROM ytempire.videos 
            WHERE created_at > NOW() - INTERVAL '24 hours'
        """)
        count = cursor.fetchone()[0]
        cursor.close()
        
        target = 500  # Target: 500 videos per day
        return {
            'count': count,
            'target': target,
            'healthy': count >= target * 0.9
        }
    
    def check_cost_per_video(self):
        """Check average cost per video"""
        cursor = self.db_conn.cursor()
        cursor.execute("""
            SELECT AVG(generation_cost) 
            FROM ytempire.videos 
            WHERE created_at > NOW() - INTERVAL '24 hours'
        """)
        avg_cost = cursor.fetchone()[0] or 0
        cursor.close()
        
        return {
            'average': float(avg_cost),
            'target': 3.0,
            'healthy': avg_cost < 3.0
        }
    
    def generate_report(self, checks):
        """Generate daily report"""
        report = f"""
        YTEMPIRE Daily Monitoring Report
        Date: {datetime.now().strftime('%Y-%m-%d')}
        
        System Health:
        - API: {'✓' if checks['api'] else '✗'}
        - Database: {'✓' if checks['database'] else '✗'}
        - Redis: {'✓' if checks['redis'] else '✗'}
        
        Video Generation:
        - Count: {checks['video_generation']['count']}/{checks['video_generation']['target']}
        - Status: {'✓' if checks['video_generation']['healthy'] else '✗'}
        
        Cost Tracking:
        - Average: ${checks['cost_tracking']['average']:.2f}
        - Target: ${checks['cost_tracking']['target']:.2f}
        - Status: {'✓' if checks['cost_tracking']['healthy'] else '✗'}
        """
        
        return report
    
    def run(self):
        """Run daily monitoring"""
        logger.info("Starting daily monitoring...")
        
        checks = self.check_system_health()
        report = self.generate_report(checks)
        
        print(report)
        
        # Send report via email/Slack
        self.send_report(report)
        
        # Clean up old data
        self.cleanup_old_data()
        
        logger.info("Daily monitoring complete!")

if __name__ == "__main__":
    monitor = DailyMonitoring()
    monitor.run()
```

### Performance Optimization

```python
#!/usr/bin/env python3
"""
performance_optimization.py - Performance tuning and optimization
"""

class PerformanceOptimizer:
    """
    Optimize system performance
    """
    
    def optimize_database(self):
        """Database optimization tasks"""
        queries = [
            # Update statistics
            "ANALYZE;",
            
            # Reindex tables
            "REINDEX TABLE ytempire.videos;",
            "REINDEX TABLE ytempire.video_metrics;",
            
            # Vacuum tables
            "VACUUM ANALYZE ytempire.videos;",
            "VACUUM ANALYZE ytempire.video_metrics;",
            
            # Update table statistics
            "UPDATE pg_statistic SET stadistinct = -1 WHERE stadistinct > 0;",
        ]
        
        for query in queries:
            self.execute_query(query)
    
    def optimize_redis(self):
        """Redis optimization"""
        # Get memory info
        info = self.redis_client.info('memory')
        
        # Check fragmentation
        if info['mem_fragmentation_ratio'] > 1.5:
            logger.warning("Redis fragmentation high, consider restart")
        
        # Clean expired keys
        self.redis_client.execute_command('MEMORY PURGE')
    
    def optimize_docker(self):
        """Docker optimization"""
        commands = [
            # Remove unused containers
            "docker container prune -f",
            
            # Remove unused images
            "docker image prune -f",
            
            # Remove unused volumes
            "docker volume prune -f",
            
            # Remove unused networks
            "docker network prune -f",
        ]
        
        for cmd in commands:
            os.system(cmd)
```

## 5.5 Incident Response

### Incident Response Playbook

```markdown
# Incident Response Playbook

## Severity Levels

### P0 - Critical
- Complete system outage
- Data loss or corruption
- Security breach
- Response time: Immediate

### P1 - High
- Major feature broken
- Significant performance degradation
- >50% of users affected
- Response time: 15 minutes

### P2 - Medium
- Minor feature broken
- Moderate performance issues
- <50% of users affected
- Response time: 1 hour

### P3 - Low
- Cosmetic issues
- Minor bugs
- No user impact
- Response time: Next business day

## Response Process

1. **Detection**
   - Automated alert triggered
   - User report received
   - Monitoring dashboard anomaly

2. **Triage**
   - Determine severity level
   - Identify affected components
   - Estimate impact scope

3. **Response**
   - Notify on-call engineer
   - Create incident channel in Slack
   - Start incident timeline

4. **Mitigation**
   - Apply immediate fix or workaround
   - Rollback if necessary
   - Monitor for stability

5. **Resolution**
   - Implement permanent fix
   - Verify system stability
   - Update documentation

6. **Post-Mortem**
   - Document root cause
   - Identify prevention measures
   - Share lessons learned
```

### Common Issues and Solutions

```yaml
# incident_responses.yaml
incidents:
  high_cpu_usage:
    symptoms:
      - CPU > 90% for 5+ minutes
      - Slow response times
      - Timeouts
    diagnosis:
      - Check top processes: "htop"
      - Check container stats: "docker stats"
      - Review logs for errors
    solutions:
      - Restart affected service
      - Scale horizontally
      - Optimize queries
      - Increase CPU allocation
  
  database_connection_pool_exhausted:
    symptoms:
      - "too many connections" errors
      - Application hangs
      - Slow queries
    diagnosis:
      - Check active connections: "SELECT count(*) FROM pg_stat_activity;"
      - Identify long-running queries
      - Check for connection leaks
    solutions:
      - Kill idle connections
      - Increase max_connections
      - Restart application
      - Fix connection leak
  
  youtube_api_quota_exceeded:
    symptoms:
      - 403 errors from YouTube API
      - Video uploads failing
      - Analytics not updating
    diagnosis:
      - Check quota usage in Google Console
      - Review API call logs
      - Identify quota-heavy operations
    solutions:
      - Switch to backup API keys
      - Enable quota pooling
      - Reduce API call frequency
      - Use YouTube Reporting API
  
  disk_space_full:
    symptoms:
      - Write operations failing
      - Database errors
      - Container crashes
    diagnosis:
      - Check disk usage: "df -h"
      - Find large files: "du -h --max-depth=1"
      - Check log sizes
    solutions:
      - Clean old logs
      - Remove old backups
      - Prune Docker resources
      - Expand disk volume
  
  memory_leak:
    symptoms:
      - Gradually increasing memory usage
      - OOM killer activating
      - Service crashes
    diagnosis:
      - Monitor memory over time
      - Profile application memory
      - Check for growing data structures
    solutions:
      - Restart affected service
      - Apply memory limits
      - Fix memory leak in code
      - Implement garbage collection
```

### Automated Recovery Scripts

```python
#!/usr/bin/env python3
"""
auto_recovery.py - Automated incident recovery
"""

import time
import subprocess
import logging
from typing import Dict, Callable

class AutoRecovery:
    """
    Automated recovery for common issues
    """
    
    def __init__(self):
        self.recovery_actions = {
            'high_cpu': self.recover_high_cpu,
            'high_memory': self.recover_high_memory,
            'service_down': self.recover_service_down,
            'database_locked': self.recover_database_locked,
            'disk_full': self.recover_disk_full
        }
    
    def detect_issues(self) -> List[str]:
        """Detect current issues"""
        issues = []
        
        # Check CPU
        cpu_usage = self.get_cpu_usage()
        if cpu_usage > 90:
            issues.append('high_cpu')
        
        # Check memory
        memory_usage = self.get_memory_usage()
        if memory_usage > 90:
            issues.append('high_memory')
        
        # Check services
        for service in ['api', 'postgres', 'redis']:
            if not self.is_service_healthy(service):
                issues.append('service_down')
        
        # Check disk space
        disk_usage = self.get_disk_usage()
        if disk_usage > 90:
            issues.append('disk_full')
        
        return issues
    
    def recover_high_cpu(self):
        """Recover from high CPU usage"""
        logger.info("Recovering from high CPU usage...")
        
        # Find and kill CPU-intensive processes
        subprocess.run(['pkill', '-9', 'ffmpeg'], check=False)
        
        # Restart API service
        subprocess.run(['docker', 'restart', 'ytempire-api'], check=True)
        
        # Clear cache
        self.redis_client.flushdb()
        
        time.sleep(30)
        
    def recover_service_down(self):
        """Recover down services"""
        logger.info("Recovering down services...")
        
        # Restart all services
        subprocess.run(['docker-compose', 'restart'], check=True)
        
        # Wait for services to come up
        time.sleep(60)
        
        # Verify health
        for service in ['api', 'postgres', 'redis']:
            if not self.is_service_healthy(service):
                logger.error(f"Service {service} still unhealthy!")
    
    def run(self):
        """Run auto-recovery"""
        while True:
            try:
                issues = self.detect_issues()
                
                for issue in issues:
                    if issue in self.recovery_actions:
                        logger.warning(f"Detected issue: {issue}")
                        self.recovery_actions[issue]()
                        logger.info(f"Recovery attempted for: {issue}")
                
                time.sleep(60)  # Check every minute
                
            except Exception as e:
                logger.error(f"Auto-recovery error: {e}")
                time.sleep(60)

if __name__ == "__main__":
    recovery = AutoRecovery()
    recovery.run()
```