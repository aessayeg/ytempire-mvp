# 4. IMPLEMENTATION GUIDES

## 4.1 Security Implementation

### MVP Security Controls

#### Week 1-2: Foundation Security
```bash
#!/bin/bash
# Initial Security Setup Script

# 1. System Hardening
apt-get update && apt-get upgrade -y
apt-get install -y ufw fail2ban unattended-upgrades

# 2. Configure Firewall
ufw default deny incoming
ufw default allow outgoing
ufw limit 22/tcp
ufw allow 80/tcp
ufw allow 443/tcp
ufw --force enable

# 3. SSH Hardening
sed -i 's/PermitRootLogin yes/PermitRootLogin no/' /etc/ssh/sshd_config
sed -i 's/#PasswordAuthentication yes/PasswordAuthentication no/' /etc/ssh/sshd_config
systemctl restart sshd

# 4. Fail2ban Configuration
cat > /etc/fail2ban/jail.local <<EOF
[DEFAULT]
bantime = 3600
findtime = 600
maxretry = 5

[sshd]
enabled = true
port = 22
EOF
systemctl enable fail2ban
systemctl start fail2ban
```

#### Week 3-4: Application Security
```python
# security/app_security.py
from functools import wraps
from flask_limiter import Limiter
import jwt
import bcrypt

class SecurityMiddleware:
    """Core security middleware for MVP"""
    
    def __init__(self, app):
        self.app = app
        self.limiter = Limiter(
            app,
            key_func=lambda: get_remote_address(),
            default_limits=["100 per hour"]
        )
        self.setup_security_headers()
    
    def setup_security_headers(self):
        @self.app.after_request
        def set_security_headers(response):
            response.headers['X-Content-Type-Options'] = 'nosniff'
            response.headers['X-Frame-Options'] = 'DENY'
            response.headers['X-XSS-Protection'] = '1; mode=block'
            response.headers['Strict-Transport-Security'] = 'max-age=31536000'
            response.headers['Content-Security-Policy'] = "default-src 'self'"
            return response
    
    def require_auth(self, f):
        @wraps(f)
        def decorated_function(*args, **kwargs):
            token = request.headers.get('Authorization')
            if not token:
                return jsonify({'error': 'No token provided'}), 401
            try:
                payload = jwt.decode(token, self.public_key, algorithms=['RS256'])
                request.user_id = payload['user_id']
            except jwt.InvalidTokenError:
                return jsonify({'error': 'Invalid token'}), 401
            return f(*args, **kwargs)
        return decorated_function
```

#### Security Checklist for MVP
```yaml
Week 1-2 Checklist:
  ✓ UFW firewall configured
  ✓ SSH key-only authentication
  ✓ Fail2ban installed
  ✓ System updates automated
  ✓ Basic monitoring setup

Week 3-4 Checklist:
  ✓ HTTPS with Let's Encrypt
  ✓ JWT authentication implemented
  ✓ Rate limiting configured
  ✓ Security headers added
  ✓ Input validation on all endpoints

Week 5-6 Checklist:
  ✓ Secrets in environment variables
  ✓ Database encryption enabled
  ✓ Backup encryption configured
  ✓ Security scanning automated
  ✓ Logging and monitoring active

Week 7-8 Checklist:
  ✓ Incident response plan documented
  ✓ Security alerts configured
  ✓ Penetration test completed
  ✓ Team security training done
  ✓ Compliance documentation ready
```

### Compliance Framework

#### MVP Compliance Requirements
```yaml
GDPR Compliance (Basic):
  User Rights:
    - Right to access data
    - Right to delete account
    - Right to data portability
    - Privacy policy published
  
  Implementation:
    - User data export endpoint
    - Account deletion workflow
    - Cookie consent banner
    - Privacy policy page
  
  Documentation:
    - Data processing registry
    - Privacy impact assessment
    - Breach notification procedure
    - Data retention policy

YouTube API Compliance:
  Requirements:
    - API key security
    - Rate limit compliance
    - Terms of service adherence
    - Branding guidelines
  
  Implementation:
    - Secure key storage
    - Rate limiting logic
    - Quota monitoring
    - Proper attribution

Payment (PCI DSS Lite):
  Scope: Using Stripe (no card data stored)
  Requirements:
    - HTTPS for all pages
    - No card data in logs
    - Secure Stripe integration
    - Regular security updates
```

### Security Tools & Automation

#### Automated Security Scanning
```yaml
# docker-compose.security.yml
version: '3.8'

services:
  vulnerability-scanner:
    image: aquasec/trivy:latest
    volumes:
      - /var/run/docker.sock:/var/run/docker.sock
    command: image --severity HIGH,CRITICAL ytempire:latest
    
  dependency-check:
    image: owasp/dependency-check:latest
    volumes:
      - ./:/src
    command: --scan /src --format JSON --out /src/reports

  security-headers:
    image: securityheaders/securityheaders:latest
    command: https://ytempire.com
```

#### Daily Security Script
```bash
#!/bin/bash
# /opt/security/daily_security_check.sh

echo "=== Daily Security Check $(date) ===" >> /var/log/security_check.log

# 1. Check for system updates
apt-get update
UPDATES=$(apt-get -s upgrade | grep -c "^Inst")
if [ "$UPDATES" -gt 0 ]; then
    echo "WARNING: $UPDATES security updates available" | mail -s "Security Updates" security@ytempire.com
fi

# 2. Scan containers for vulnerabilities
docker run --rm -v /var/run/docker.sock:/var/run/docker.sock \
    aquasec/trivy image --severity HIGH,CRITICAL $(docker ps --format "{{.Image}}")

# 3. Check SSL certificate expiry
CERT_EXPIRY=$(echo | openssl s_client -connect ytempire.com:443 2>/dev/null | \
    openssl x509 -noout -enddate | cut -d= -f2)
echo "SSL Certificate expires: $CERT_EXPIRY"

# 4. Review failed login attempts
grep "Failed password" /var/log/auth.log | tail -20

# 5. Check disk usage for logs
du -sh /var/log/* | sort -rh | head -10

# 6. Verify backup completion
if [ -f /backup/daily_$(date +%Y%m%d).tar.gz.enc ]; then
    echo "Backup completed successfully"
else
    echo "CRITICAL: Backup failed!" | mail -s "Backup Failure" ops@ytempire.com
fi
```

### Secrets Management

#### MVP Secrets Strategy (Environment Variables)
```bash
# /etc/ytempire/.env.encrypted
# Encrypted with age (https://github.com/FiloSottile/age)

# Database
DATABASE_URL=postgresql://user:pass@localhost/ytempire
REDIS_URL=redis://localhost:6379/0

# External APIs
OPENAI_API_KEY=sk-...
ELEVENLABS_API_KEY=...
YOUTUBE_CLIENT_ID=...
YOUTUBE_CLIENT_SECRET=...
STRIPE_SECRET_KEY=sk_live_...

# Security
JWT_PRIVATE_KEY_PATH=/etc/ytempire/keys/jwt_private.pem
JWT_PUBLIC_KEY_PATH=/etc/ytempire/keys/jwt_public.pem
ENCRYPTION_KEY=...

# Monitoring
SENTRY_DSN=...
```

#### Secrets Rotation Procedure
```yaml
Monthly Rotation:
  API Keys:
    1. Generate new key in provider dashboard
    2. Update staging environment
    3. Test all integrations
    4. Update production
    5. Revoke old key after 24 hours
  
  Database Passwords:
    1. Create new user with same permissions
    2. Update application config
    3. Test connections
    4. Remove old user
  
  JWT Keys:
    1. Generate new key pair
    2. Deploy with both keys active
    3. Wait for token expiry (7 days)
    4. Remove old keys
```

---

## 4.2 Platform Operations

### Deployment Procedures

#### Daily Deployment Process
```yaml
Deployment Window: 8:00 PM - 10:00 PM (low traffic)
Frequency: Maximum once per day
Type: Blue-Green deployment

Pre-Deployment Checklist:
  ✓ All tests passing (>95%)
  ✓ Security scan completed
  ✓ Database migrations tested
  ✓ Rollback plan prepared
  ✓ Team notification sent

Deployment Steps:
  1. Create backup snapshot
  2. Pull latest code
  3. Build Docker images
  4. Run database migrations
  5. Deploy to staging container
  6. Run smoke tests
  7. Switch traffic to new container
  8. Monitor for 30 minutes
  9. Remove old container
```

#### Deployment Script
```bash
#!/bin/bash
# /opt/deployment/deploy.sh

set -e

# Configuration
REPO_DIR="/opt/ytempire"
BACKUP_DIR="/backup"
DOCKER_COMPOSE="/usr/local/bin/docker-compose"

echo "Starting deployment at $(date)"

# 1. Backup current state
echo "Creating backup..."
$DOCKER_COMPOSE exec postgres pg_dump -U ytempire > $BACKUP_DIR/pre_deploy_$(date +%Y%m%d_%H%M%S).sql

# 2. Pull latest code
echo "Pulling latest code..."
cd $REPO_DIR
git pull origin main

# 3. Build new images
echo "Building Docker images..."
$DOCKER_COMPOSE build

# 4. Deploy with zero downtime
echo "Deploying new version..."
$DOCKER_COMPOSE up -d --no-deps --build backend
$DOCKER_COMPOSE up -d --no-deps --build frontend

# 5. Health check
echo "Running health checks..."
sleep 10
curl -f http://localhost:8000/health || exit 1

# 6. Run migrations
echo "Running database migrations..."
$DOCKER_COMPOSE exec backend python manage.py migrate

echo "Deployment completed successfully at $(date)"

# 7. Notify team
curl -X POST https://hooks.slack.com/services/xxx \
  -H 'Content-Type: application/json' \
  -d '{"text":"Deployment completed successfully"}'
```

### Monitoring & Observability

#### Prometheus Configuration
```yaml
# prometheus.yml
global:
  scrape_interval: 30s
  evaluation_interval: 30s

scrape_configs:
  - job_name: 'backend'
    static_configs:
      - targets: ['backend:8001']
    metrics_path: /metrics
  
  - job_name: 'postgres'
    static_configs:
      - targets: ['postgres-exporter:9187']
  
  - job_name: 'redis'
    static_configs:
      - targets: ['redis-exporter:9121']
  
  - job_name: 'node'
    static_configs:
      - targets: ['node-exporter:9100']

alerting:
  alertmanagers:
    - static_configs:
        - targets: ['alertmanager:9093']

rule_files:
  - '/etc/prometheus/alerts.yml'
```

#### Alert Rules
```yaml
# alerts.yml
groups:
  - name: critical
    rules:
      - alert: HighCPUUsage
        expr: node_cpu_usage > 90
        for: 5m
        labels:
          severity: critical
        annotations:
          summary: "High CPU usage detected"
      
      - alert: LowDiskSpace
        expr: node_filesystem_free_bytes / node_filesystem_size_bytes < 0.1
        for: 5m
        labels:
          severity: warning
        annotations:
          summary: "Less than 10% disk space remaining"
      
      - alert: ServiceDown
        expr: up == 0
        for: 1m
        labels:
          severity: critical
        annotations:
          summary: "Service {{ $labels.job }} is down"
```

### CI/CD Pipeline

#### GitHub Actions Configuration
```yaml
# .github/workflows/deploy.yml
name: Deploy to Production

on:
  push:
    branches: [main]

jobs:
  test:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v2
      
      - name: Run tests
        run: |
          docker-compose -f docker-compose.test.yml up --abort-on-container-exit
          docker-compose -f docker-compose.test.yml down
      
      - name: Security scan
        run: |
          docker run --rm -v "$PWD":/src \
            aquasec/trivy fs --severity HIGH,CRITICAL /src
  
  deploy:
    needs: test
    runs-on: self-hosted
    if: github.ref == 'refs/heads/main'
    steps:
      - name: Deploy to server
        run: |
          ssh ytempire@server "cd /opt/ytempire && ./deploy.sh"
      
      - name: Verify deployment
        run: |
          sleep 30
          curl -f https://ytempire.com/health || exit 1
```

### Disaster Recovery

#### Backup Strategy
```yaml
Backup Configuration:
  Database:
    Type: PostgreSQL pg_dump
    Frequency: Every 6 hours
    Retention: 30 days
    Storage: External drives (2 copies)
  
  Files:
    Type: Rsync + tar
    Frequency: Daily
    Retention: 7 days
    Includes:
      - Application code
      - Configuration files
      - Media assets
      - Logs
  
  Verification:
    Test Restore: Weekly
    Integrity Check: Daily (SHA-256)
    Documentation: Restore procedures
```

#### Recovery Procedures
```bash
#!/bin/bash
# /opt/recovery/disaster_recovery.sh

echo "=== Disaster Recovery Procedure ==="

# 1. Stop all services
docker-compose down

# 2. Restore database
echo "Restoring database..."
LATEST_BACKUP=$(ls -t /backup/postgres/*.sql.gz | head -1)
gunzip < $LATEST_BACKUP | docker exec -i postgres psql -U ytempire

# 3. Restore application files
echo "Restoring application files..."
LATEST_APP_BACKUP=$(ls -t /backup/app/*.tar.gz | head -1)
tar -xzf $LATEST_APP_BACKUP -C /opt/ytempire

# 4. Restore media files
echo "Restoring media files..."
LATEST_MEDIA_BACKUP=$(ls -t /backup/media/*.tar.gz | head -1)
tar -xzf $LATEST_MEDIA_BACKUP -C /opt/ytempire/media

# 5. Start services
docker-compose up -d

# 6. Verify services
sleep 30
docker-compose ps
curl -f http://localhost:8000/health

echo "Recovery completed. Please verify all services."
```

---

## 4.3 Development Guidelines

### Security Coding Standards

#### Input Validation
```python
# validators.py
from pydantic import BaseModel, validator
import re

class VideoGenerationRequest(BaseModel):
    title: str
    description: str
    channel_id: str
    
    @validator('title')
    def validate_title(cls, v):
        if len(v) > 100:
            raise ValueError('Title too long')
        if not re.match(r'^[a-zA-Z0-9\s\-\_]+$', v):
            raise ValueError('Invalid characters in title')
        return v
    
    @validator('channel_id')
    def validate_channel_id(cls, v):
        if not re.match(r'^[a-f0-9\-]{36}$', v):
            raise ValueError('Invalid channel ID format')
        return v
```

#### SQL Injection Prevention
```python
# database.py
from sqlalchemy import text

# NEVER DO THIS
def bad_query(user_input):
    query = f"SELECT * FROM users WHERE email = '{user_input}'"
    return db.execute(query)

# DO THIS INSTEAD
def safe_query(user_input):
    query = text("SELECT * FROM users WHERE email = :email")
    return db.execute(query, {"email": user_input})
```

#### XSS Prevention
```javascript
// frontend/utils/sanitize.js

// NEVER DO THIS
function dangerousRender(userContent) {
    element.innerHTML = userContent;
}

// DO THIS INSTEAD
function safeRender(userContent) {
    element.textContent = userContent;
    // Or use a library like DOMPurify
    element.innerHTML = DOMPurify.sanitize(userContent);
}
```

### Testing Requirements

#### Test Coverage Targets
```yaml
MVP Testing Requirements:
  Unit Tests:
    Coverage: 70% minimum
    Critical Paths: 100%
    Run Time: <5 minutes
  
  Integration Tests:
    API Endpoints: All authenticated endpoints
    Database: All CRUD operations
    External APIs: Mocked responses
  
  Security Tests:
    Authentication: All flows tested
    Authorization: Permission matrix
    Input Validation: Fuzzing tests
    SQL Injection: Parameterized queries
```

#### Test Examples
```python
# test_security.py
import pytest
from app import create_app

class TestSecurity:
    def test_jwt_required(self, client):
        """Test that endpoints require authentication"""
        response = client.get('/api/channels')
        assert response.status_code == 401
    
    def test_sql_injection(self, client, auth_headers):
        """Test SQL injection prevention"""
        malicious_input = "'; DROP TABLE users; --"
        response = client.get(
            f'/api/search?q={malicious_input}',
            headers=auth_headers
        )
        assert response.status_code == 200
        # Verify tables still exist
    
    def test_rate_limiting(self, client, auth_headers):
        """Test rate limiting works"""
        for i in range(101):
            response = client.get('/api/channels', headers=auth_headers)
        assert response.status_code == 429
```

---

## Document Metadata

**Version**: 2.0  
**Last Updated**: January 2025  
**Owner**: Platform Operations Lead  
**Review Cycle**: Sprint Review  
**Distribution**: All Technical Teams  

**Key Consolidations**:
- Merged security implementation guides
- Unified deployment procedures
- Consolidated monitoring configuration
- Added disaster recovery procedures
- Included development security guidelines