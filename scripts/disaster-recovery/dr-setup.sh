#!/bin/bash

# YTEmpire Disaster Recovery Setup Script
# Sets up comprehensive backup and monitoring infrastructure

set -euo pipefail

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Configuration
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
BACKUP_DIR="/var/backups/ytempire"
LOG_FILE="/var/log/ytempire-dr-setup.log"

# Logging function
log() {
    echo "$(date '+%Y-%m-%d %H:%M:%S') - $1" | tee -a "$LOG_FILE"
}

error() {
    echo -e "${RED}ERROR: $1${NC}" | tee -a "$LOG_FILE"
    exit 1
}

warning() {
    echo -e "${YELLOW}WARNING: $1${NC}" | tee -a "$LOG_FILE"
}

success() {
    echo -e "${GREEN}SUCCESS: $1${NC}" | tee -a "$LOG_FILE"
}

# Check if running as root
check_root() {
    if [[ $EUID -eq 0 ]]; then
        error "This script should not be run as root for security reasons"
    fi
}

# Check prerequisites
check_prerequisites() {
    log "Checking prerequisites..."
    
    # Check Docker
    if ! command -v docker &> /dev/null; then
        error "Docker is not installed. Please install Docker first."
    fi
    
    # Check Docker Compose
    if ! command -v docker-compose &> /dev/null; then
        error "Docker Compose is not installed. Please install Docker Compose first."
    fi
    
    # Check AWS CLI (optional but recommended)
    if ! command -v aws &> /dev/null; then
        warning "AWS CLI is not installed. S3 backups may not work properly."
    fi
    
    # Check disk space
    AVAILABLE_SPACE=$(df "$PROJECT_ROOT" | tail -1 | awk '{print $4}')
    REQUIRED_SPACE=10485760  # 10GB in KB
    
    if [[ $AVAILABLE_SPACE -lt $REQUIRED_SPACE ]]; then
        warning "Less than 10GB available disk space. Consider freeing up space."
    fi
    
    success "Prerequisites check completed"
}

# Create necessary directories
create_directories() {
    log "Creating necessary directories..."
    
    # Create backup directory
    sudo mkdir -p "$BACKUP_DIR"
    sudo chown "$(whoami):$(whoami)" "$BACKUP_DIR"
    
    # Create monitoring directories
    mkdir -p "$PROJECT_ROOT/infrastructure/monitoring/grafana/provisioning/dashboards"
    mkdir -p "$PROJECT_ROOT/infrastructure/monitoring/grafana/provisioning/datasources"
    mkdir -p "$PROJECT_ROOT/infrastructure/monitoring/grafana/dashboards"
    
    # Create log directories
    sudo mkdir -p /var/log/ytempire
    sudo chown "$(whoami):$(whoami)" /var/log/ytempire
    
    success "Directories created"
}

# Setup environment variables
setup_environment() {
    log "Setting up environment variables..."
    
    ENV_FILE="$PROJECT_ROOT/.env.disaster-recovery"
    
    if [[ ! -f "$ENV_FILE" ]]; then
        cat > "$ENV_FILE" << EOF
# Disaster Recovery Environment Variables

# Database Configuration
DATABASE_PASSWORD=secure_postgres_password_$(openssl rand -hex 8)
REPLICATION_PASSWORD=secure_replication_password_$(openssl rand -hex 8)

# Redis Configuration
REDIS_PASSWORD=secure_redis_password_$(openssl rand -hex 8)

# MinIO Configuration
MINIO_ACCESS_KEY=ytempire_access
MINIO_SECRET_KEY=ytempire_secret_key_$(openssl rand -hex 16)

# AWS Configuration (for S3 backups)
AWS_ACCESS_KEY_ID=your_aws_access_key_here
AWS_SECRET_ACCESS_KEY=your_aws_secret_key_here
AWS_REGION=us-east-1
BACKUP_S3_BUCKET=ytempire-backups-$(openssl rand -hex 4)

# Backup Configuration
BACKUP_ENCRYPTION_KEY=$(openssl rand -hex 32)

# Monitoring Configuration
GRAFANA_PASSWORD=admin_$(openssl rand -hex 8)

# Webhook URLs (optional)
BACKUP_WEBHOOK_URL=
SLACK_WEBHOOK_URL=

# Alert Configuration
ALERT_EMAIL=admin@yourdomain.com
SMTP_SERVER=smtp.gmail.com
SMTP_PORT=587
SMTP_USER=your_email@gmail.com
SMTP_PASSWORD=your_email_password
EOF

        success "Environment file created: $ENV_FILE"
        warning "Please review and update the environment variables in $ENV_FILE"
    else
        log "Environment file already exists: $ENV_FILE"
    fi
}

# Generate monitoring configuration
generate_monitoring_config() {
    log "Generating monitoring configuration..."
    
    # Prometheus configuration
    cat > "$PROJECT_ROOT/infrastructure/monitoring/prometheus.yml" << 'EOF'
global:
  scrape_interval: 15s
  evaluation_interval: 15s

alerting:
  alertmanagers:
    - static_configs:
        - targets:
          - alertmanager:9093

rule_files:
  - "alert_rules.yml"

scrape_configs:
  - job_name: 'prometheus'
    static_configs:
      - targets: ['localhost:9090']

  - job_name: 'node-exporter'
    static_configs:
      - targets: ['node_exporter:9100']

  - job_name: 'cadvisor'
    static_configs:
      - targets: ['cadvisor:8080']

  - job_name: 'postgres-exporter'
    static_configs:
      - targets: ['postgres_exporter:9187']

  - job_name: 'redis-exporter'
    static_configs:
      - targets: ['redis_exporter:9121']

  - job_name: 'ytempire-backend'
    static_configs:
      - targets: ['backend:8000']
    metrics_path: '/metrics'

  - job_name: 'ytempire-health'
    static_configs:
      - targets: ['health_monitor:8000']
    metrics_path: '/health'
    scrape_interval: 30s
EOF

    # AlertManager configuration
    cat > "$PROJECT_ROOT/infrastructure/monitoring/alertmanager.yml" << 'EOF'
global:
  smtp_smarthost: 'smtp.gmail.com:587'
  smtp_from: 'alerts@ytempire.com'
  smtp_auth_username: 'your_email@gmail.com'
  smtp_auth_password: 'your_email_password'

route:
  group_by: ['alertname']
  group_wait: 10s
  group_interval: 10s
  repeat_interval: 1h
  receiver: 'web.hook'

receivers:
- name: 'web.hook'
  email_configs:
  - to: 'admin@yourdomain.com'
    subject: 'YTEmpire Alert: {{ .GroupLabels.alertname }}'
    body: |
      {{ range .Alerts }}
      Alert: {{ .Annotations.summary }}
      Description: {{ .Annotations.description }}
      {{ end }}

inhibit_rules:
  - source_match:
      severity: 'critical'
    target_match:
      severity: 'warning'
    equal: ['alertname', 'dev', 'instance']
EOF

    # Loki configuration
    cat > "$PROJECT_ROOT/infrastructure/monitoring/loki-config.yml" << 'EOF'
auth_enabled: false

server:
  http_listen_port: 3100
  grpc_listen_port: 9096

common:
  path_prefix: /loki
  storage:
    filesystem:
      chunks_directory: /loki/chunks
      rules_directory: /loki/rules
  replication_factor: 1
  ring:
    instance_addr: 127.0.0.1
    kvstore:
      store: inmemory

query_scheduler:
  max_outstanding_requests_per_tenant: 1024

schema_config:
  configs:
    - from: 2020-10-24
      store: boltdb-shipper
      object_store: filesystem
      schema: v11
      index:
        prefix: index_
        period: 24h

ruler:
  alertmanager_url: http://localhost:9093

analytics:
  reporting_enabled: false
EOF

    # Promtail configuration
    cat > "$PROJECT_ROOT/infrastructure/monitoring/promtail-config.yml" << 'EOF'
server:
  http_listen_port: 9080
  grpc_listen_port: 0

positions:
  filename: /tmp/positions.yaml

clients:
  - url: http://loki:3100/loki/api/v1/push

scrape_configs:
- job_name: system
  static_configs:
  - targets:
      - localhost
    labels:
      job: varlogs
      __path__: /var/log/*log

- job_name: containers
  static_configs:
  - targets:
      - localhost
    labels:
      job: containerlogs
      __path__: /var/lib/docker/containers/*/*log
  pipeline_stages:
  - json:
      expressions:
        output: log
        stream: stream
        attrs:
  - json:
      expressions:
        tag:
      source: attrs
  - regex:
      expression: (?P<container_name>(?:[^|]*))\|
      source: tag
  - timestamp:
      format: RFC3339Nano
      source: time
  - labels:
      stream:
      container_name:
  - output:
      source: output
EOF

    # Grafana datasource provisioning
    cat > "$PROJECT_ROOT/infrastructure/monitoring/grafana/provisioning/datasources/datasources.yml" << 'EOF'
apiVersion: 1

datasources:
  - name: Prometheus
    type: prometheus
    access: proxy
    url: http://prometheus:9090
    isDefault: true

  - name: Loki
    type: loki
    access: proxy
    url: http://loki:3100
EOF

    success "Monitoring configuration generated"
}

# Setup backup scripts
setup_backup_scripts() {
    log "Setting up backup scripts..."
    
    # Create systemd service for backup scheduler (if not using Docker)
    if command -v systemctl &> /dev/null; then
        cat > "$PROJECT_ROOT/scripts/ytempire-backup.service" << EOF
[Unit]
Description=YTEmpire Backup Service
After=network.target

[Service]
Type=simple
User=$(whoami)
WorkingDirectory=$PROJECT_ROOT
ExecStart=/usr/bin/python3 $PROJECT_ROOT/infrastructure/backup/backup_manager.py schedule
Restart=always
RestartSec=30

[Install]
WantedBy=multi-user.target
EOF

        warning "Systemd service file created at $PROJECT_ROOT/scripts/ytempire-backup.service"
        warning "To install: sudo cp $PROJECT_ROOT/scripts/ytempire-backup.service /etc/systemd/system/"
        warning "Then run: sudo systemctl enable ytempire-backup && sudo systemctl start ytempire-backup"
    fi
    
    # Create backup restoration script
    cat > "$PROJECT_ROOT/scripts/restore-backup.sh" << 'EOF'
#!/bin/bash

# YTEmpire Backup Restoration Script

set -euo pipefail

BACKUP_ID="$1"
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

if [[ -z "$BACKUP_ID" ]]; then
    echo "Usage: $0 <backup_id>"
    echo "Example: $0 full_20240101_120000"
    exit 1
fi

echo "Starting restoration from backup: $BACKUP_ID"

# Stop services
echo "Stopping services..."
cd "$PROJECT_ROOT"
docker-compose -f docker-compose.disaster-recovery.yml stop backend postgres_primary redis_primary

# Run restoration
echo "Running restoration..."
docker-compose -f docker-compose.disaster-recovery.yml run --rm backup_manager \
    python /app/backup/backup_manager.py restore --backup-id "$BACKUP_ID"

# Start services
echo "Starting services..."
docker-compose -f docker-compose.disaster-recovery.yml up -d

echo "Restoration completed!"
EOF

    chmod +x "$PROJECT_ROOT/scripts/restore-backup.sh"
    
    success "Backup scripts setup completed"
}

# Validate Docker Compose configuration
validate_docker_compose() {
    log "Validating Docker Compose configuration..."
    
    cd "$PROJECT_ROOT"
    
    if docker-compose -f docker-compose.disaster-recovery.yml config > /dev/null; then
        success "Docker Compose configuration is valid"
    else
        error "Docker Compose configuration validation failed"
    fi
}

# Test backup system
test_backup_system() {
    log "Testing backup system..."
    
    cd "$PROJECT_ROOT"
    
    # Start services
    echo "Starting disaster recovery services..."
    docker-compose -f docker-compose.disaster-recovery.yml up -d postgres_primary redis_primary backup_manager
    
    # Wait for services to be ready
    sleep 30
    
    # Test database connection
    if docker-compose -f docker-compose.disaster-recovery.yml exec -T postgres_primary pg_isready -U postgres; then
        success "Database connection test passed"
    else
        error "Database connection test failed"
    fi
    
    # Test Redis connection
    if docker-compose -f docker-compose.disaster-recovery.yml exec -T redis_primary redis-cli ping; then
        success "Redis connection test passed"
    else
        error "Redis connection test failed"
    fi
    
    # Test backup creation (dry run)
    echo "Testing backup creation..."
    docker-compose -f docker-compose.disaster-recovery.yml exec -T backup_manager \
        python /app/backup/backup_manager.py backup --type full
    
    success "Backup system test completed"
}

# Print final instructions
print_instructions() {
    log "Setup completed successfully!"
    
    cat << EOF

${GREEN}YTEmpire Disaster Recovery Setup Complete!${NC}

${YELLOW}Next Steps:${NC}

1. Review and update environment variables:
   ${PROJECT_ROOT}/.env.disaster-recovery

2. Start the disaster recovery stack:
   cd ${PROJECT_ROOT}
   docker-compose -f docker-compose.disaster-recovery.yml up -d

3. Access monitoring dashboards:
   - Grafana: http://localhost:3001 (admin / check .env file for password)
   - Prometheus: http://localhost:9090
   - MinIO Console: http://localhost:9001

4. Test backup and restore:
   # Create backup
   docker-compose -f docker-compose.disaster-recovery.yml exec backup_manager \
     python /app/backup/backup_manager.py backup --type full
   
   # List backups
   docker-compose -f docker-compose.disaster-recovery.yml exec backup_manager \
     python /app/backup/backup_manager.py status
   
   # Restore backup
   ./scripts/restore-backup.sh <backup_id>

5. Set up monitoring alerts:
   - Update AlertManager configuration with your email settings
   - Configure Slack webhooks for notifications

6. Schedule regular backups:
   - Full backups run every 24 hours by default
   - Incremental backups run every 4 hours by default
   - Customize in backup_manager.py or environment variables

${YELLOW}Important Files:${NC}
- Environment: ${PROJECT_ROOT}/.env.disaster-recovery
- Docker Compose: ${PROJECT_ROOT}/docker-compose.disaster-recovery.yml
- Backup Manager: ${PROJECT_ROOT}/infrastructure/backup/backup_manager.py
- Health Monitor: ${PROJECT_ROOT}/infrastructure/monitoring/health_check.py
- Logs: /var/log/ytempire/

${YELLOW}Recovery Procedures:${NC}
- RTO (Recovery Time Objective): 4 hours
- RPO (Recovery Point Objective): 1 hour
- See documentation in _documentation/Platform OPS TL/ for detailed procedures

EOF
}

# Main execution
main() {
    log "Starting YTEmpire Disaster Recovery Setup"
    
    check_root
    check_prerequisites
    create_directories
    setup_environment
    generate_monitoring_config
    setup_backup_scripts
    validate_docker_compose
    
    # Optional: Run tests if --test flag provided
    if [[ "${1:-}" == "--test" ]]; then
        test_backup_system
    fi
    
    print_instructions
    
    success "Disaster Recovery setup completed successfully!"
}

# Run main function with all arguments
main "$@"