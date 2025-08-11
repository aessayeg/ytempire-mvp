#!/bin/bash

# YTEmpire Auto-Scaling Setup Script
# Sets up comprehensive auto-scaling infrastructure

set -euo pipefail

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Configuration
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
LOG_FILE="/var/log/ytempire-scaling-setup.log"

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

# Check prerequisites
check_prerequisites() {
    log "Checking prerequisites for auto-scaling setup..."
    
    # Check Docker
    if ! command -v docker &> /dev/null; then
        error "Docker is not installed. Please install Docker first."
    fi
    
    # Check Docker Compose
    if ! command -v docker-compose &> /dev/null; then
        error "Docker Compose is not installed. Please install Docker Compose first."
    fi
    
    # Check if Docker daemon is running
    if ! docker info &> /dev/null; then
        error "Docker daemon is not running. Please start Docker."
    fi
    
    # Check available memory
    AVAILABLE_MEMORY=$(free -m | awk 'NR==2{print $7}')
    if [[ $AVAILABLE_MEMORY -lt 4096 ]]; then
        warning "Less than 4GB available memory. Auto-scaling may be limited."
    fi
    
    # Check for GPU (optional)
    if command -v nvidia-smi &> /dev/null; then
        success "NVIDIA GPU detected - GPU scaling enabled"
    else
        warning "No NVIDIA GPU detected - GPU scaling disabled"
    fi
    
    success "Prerequisites check completed"
}

# Setup environment variables
setup_environment() {
    log "Setting up scaling environment variables..."
    
    ENV_FILE="$PROJECT_ROOT/.env.scaling"
    
    if [[ ! -f "$ENV_FILE" ]]; then
        cat > "$ENV_FILE" << EOF
# Auto-Scaling Environment Variables

# Database Configuration
DATABASE_PASSWORD=secure_postgres_password_$(openssl rand -hex 8)

# Redis Configuration  
REDIS_PASSWORD=secure_redis_password_$(openssl rand -hex 8)

# JWT Configuration
JWT_SECRET_KEY=secure_jwt_secret_$(openssl rand -hex 32)

# Scaling Configuration
SCALING_ENABLED=true
SCALING_INTERVAL=30
MAX_BACKEND_INSTANCES=10
MAX_WORKER_INSTANCES=20
MIN_BACKEND_INSTANCES=2
MIN_WORKER_INSTANCES=1

# CPU and Memory Thresholds
CPU_SCALE_UP_THRESHOLD=80
CPU_SCALE_DOWN_THRESHOLD=30
MEMORY_SCALE_UP_THRESHOLD=85
MEMORY_SCALE_DOWN_THRESHOLD=50

# Queue Thresholds
QUEUE_SCALE_UP_THRESHOLD=50
QUEUE_SCALE_DOWN_THRESHOLD=10

# Response Time Thresholds (milliseconds)
RESPONSE_TIME_SCALE_UP_THRESHOLD=2000
RESPONSE_TIME_SCALE_DOWN_THRESHOLD=500

# Load Balancer Configuration
HAPROXY_STATS_PASSWORD=stats_$(openssl rand -hex 8)

# Monitoring Configuration
GRAFANA_PASSWORD=admin_$(openssl rand -hex 8)
PROMETHEUS_RETENTION=30d

# Orchestrator (docker or kubernetes)
ORCHESTRATOR=docker

# Resource Limits
BACKEND_CPU_LIMIT=2
BACKEND_MEMORY_LIMIT=2G
WORKER_CPU_LIMIT=4
WORKER_MEMORY_LIMIT=8G

# Logging Level
LOG_LEVEL=INFO
EOF

        success "Environment file created: $ENV_FILE"
        warning "Please review and update the environment variables in $ENV_FILE"
    else
        log "Environment file already exists: $ENV_FILE"
    fi
}

# Create scaling configuration
create_scaling_config() {
    log "Creating scaling configuration files..."
    
    # Create scaling configuration JSON
    cat > "$PROJECT_ROOT/infrastructure/scaling/scaling-config.json" << 'EOF'
{
  "backend": {
    "name": "backend",
    "current_instances": 2,
    "min_instances": 1,
    "max_instances": 10,
    "cpu_request": "500m",
    "cpu_limit": "2",
    "memory_request": "512Mi", 
    "memory_limit": "2Gi",
    "rules": [
      {
        "name": "cpu_usage",
        "metric": "cpu_usage",
        "threshold_up": 80.0,
        "threshold_down": 30.0,
        "action": "horizontal",
        "cooldown_seconds": 300,
        "min_instances": 1,
        "max_instances": 10,
        "scale_factor": 1.5,
        "enabled": true
      },
      {
        "name": "response_time",
        "metric": "response_time_p95", 
        "threshold_up": 2000.0,
        "threshold_down": 500.0,
        "action": "horizontal",
        "cooldown_seconds": 180,
        "min_instances": 1,
        "max_instances": 8,
        "scale_factor": 1.3,
        "enabled": true
      },
      {
        "name": "queue_depth",
        "metric": "queue_depth",
        "threshold_up": 50,
        "threshold_down": 10,
        "action": "horizontal", 
        "cooldown_seconds": 120,
        "min_instances": 1,
        "max_instances": 15,
        "scale_factor": 2.0,
        "enabled": true
      }
    ]
  },
  "celery_worker": {
    "name": "celery_worker",
    "current_instances": 3,
    "min_instances": 1,
    "max_instances": 20,
    "cpu_request": "1",
    "cpu_limit": "4",
    "memory_request": "1Gi",
    "memory_limit": "8Gi", 
    "rules": [
      {
        "name": "video_queue",
        "metric": "video_queue_length",
        "threshold_up": 20,
        "threshold_down": 5,
        "action": "horizontal",
        "cooldown_seconds": 180,
        "min_instances": 1,
        "max_instances": 20,
        "scale_factor": 1.5,
        "enabled": true
      },
      {
        "name": "gpu_utilization",
        "metric": "gpu_utilization",
        "threshold_up": 85.0,
        "threshold_down": 40.0,
        "action": "horizontal",
        "cooldown_seconds": 300,
        "min_instances": 1,
        "max_instances": 10,
        "scale_factor": 1.2,
        "enabled": true
      }
    ]
  },
  "redis": {
    "name": "redis",
    "current_instances": 1,
    "min_instances": 1,
    "max_instances": 3,
    "cpu_request": "250m",
    "cpu_limit": "1",
    "memory_request": "256Mi",
    "memory_limit": "1Gi",
    "rules": [
      {
        "name": "memory_usage", 
        "metric": "memory_usage",
        "threshold_up": 80.0,
        "threshold_down": 50.0,
        "action": "vertical",
        "cooldown_seconds": 600,
        "min_instances": 1,
        "max_instances": 3,
        "scale_factor": 1.5,
        "enabled": true
      }
    ]
  }
}
EOF

    # Create Prometheus scaling rules
    cat > "$PROJECT_ROOT/infrastructure/scaling/scaling-rules.yml" << 'EOF'
groups:
- name: scaling_rules
  rules:
  # CPU usage rules
  - alert: HighCPUUsage
    expr: rate(node_cpu_seconds_total{mode!="idle"}[5m]) * 100 > 80
    for: 5m
    labels:
      severity: warning
      component: system
      action: scale_up
    annotations:
      summary: "High CPU usage detected"
      description: "CPU usage is above 80% for more than 5 minutes"

  - alert: LowCPUUsage
    expr: rate(node_cpu_seconds_total{mode!="idle"}[5m]) * 100 < 30
    for: 10m
    labels:
      severity: info
      component: system
      action: scale_down
    annotations:
      summary: "Low CPU usage detected"
      description: "CPU usage is below 30% for more than 10 minutes"

  # Memory usage rules
  - alert: HighMemoryUsage
    expr: (1 - (node_memory_MemAvailable_bytes / node_memory_MemTotal_bytes)) * 100 > 85
    for: 5m
    labels:
      severity: warning
      component: system
      action: scale_up
    annotations:
      summary: "High memory usage detected"
      description: "Memory usage is above 85% for more than 5 minutes"

  # Queue depth rules
  - alert: HighQueueDepth
    expr: video_queue_depth > 50
    for: 2m
    labels:
      severity: warning
      component: application
      action: scale_up
    annotations:
      summary: "High queue depth detected"
      description: "Video queue depth is above 50 items"

  # Response time rules
  - alert: HighResponseTime
    expr: histogram_quantile(0.95, rate(http_request_duration_seconds_bucket[5m])) > 2
    for: 3m
    labels:
      severity: warning
      component: application
      action: scale_up
    annotations:
      summary: "High response time detected"
      description: "95th percentile response time is above 2 seconds"

  # GPU utilization rules (if available)
  - alert: HighGPUUtilization
    expr: gpu_utilization_percent > 85
    for: 5m
    labels:
      severity: warning
      component: gpu
      action: scale_up
    annotations:
      summary: "High GPU utilization detected"
      description: "GPU utilization is above 85%"
EOF

    success "Scaling configuration created"
}

# Setup monitoring dashboards
setup_monitoring_dashboards() {
    log "Setting up monitoring dashboards for scaling..."
    
    # Create Grafana dashboard for scaling
    mkdir -p "$PROJECT_ROOT/infrastructure/scaling/grafana/dashboards"
    
    cat > "$PROJECT_ROOT/infrastructure/scaling/grafana/dashboards/scaling-dashboard.json" << 'EOF'
{
  "dashboard": {
    "id": null,
    "title": "YTEmpire Auto-Scaling Dashboard",
    "tags": ["ytempire", "scaling", "auto-scaling"],
    "style": "dark",
    "timezone": "browser",
    "refresh": "30s",
    "time": {
      "from": "now-1h",
      "to": "now"
    },
    "panels": [
      {
        "id": 1,
        "title": "Service Instance Count",
        "type": "stat",
        "targets": [
          {
            "expr": "up{job=\"ytempire-backend\"}",
            "legendFormat": "Backend Instances"
          },
          {
            "expr": "up{job=\"ytempire-celery\"}",
            "legendFormat": "Worker Instances"
          }
        ],
        "fieldConfig": {
          "defaults": {
            "color": {
              "mode": "palette-classic"
            },
            "thresholds": {
              "steps": [
                {"color": "green", "value": null},
                {"color": "red", "value": 80}
              ]
            }
          }
        }
      },
      {
        "id": 2,
        "title": "CPU Usage",
        "type": "timeseries",
        "targets": [
          {
            "expr": "rate(node_cpu_seconds_total{mode!=\"idle\"}[5m]) * 100",
            "legendFormat": "CPU Usage %"
          }
        ]
      },
      {
        "id": 3,
        "title": "Memory Usage", 
        "type": "timeseries",
        "targets": [
          {
            "expr": "(1 - (node_memory_MemAvailable_bytes / node_memory_MemTotal_bytes)) * 100",
            "legendFormat": "Memory Usage %"
          }
        ]
      },
      {
        "id": 4,
        "title": "Video Queue Depth",
        "type": "timeseries",
        "targets": [
          {
            "expr": "video_queue_depth",
            "legendFormat": "Queue Depth"
          }
        ]
      },
      {
        "id": 5,
        "title": "Response Time P95",
        "type": "timeseries",
        "targets": [
          {
            "expr": "histogram_quantile(0.95, rate(http_request_duration_seconds_bucket[5m]))",
            "legendFormat": "Response Time P95"
          }
        ]
      },
      {
        "id": 6,
        "title": "Scaling Events",
        "type": "table",
        "targets": [
          {
            "expr": "increase(scaling_events_total[1h])",
            "legendFormat": "Scaling Events"
          }
        ]
      }
    ]
  }
}
EOF

    # Create datasource configuration
    mkdir -p "$PROJECT_ROOT/infrastructure/scaling/grafana/provisioning/datasources"
    
    cat > "$PROJECT_ROOT/infrastructure/scaling/grafana/provisioning/datasources/prometheus.yml" << 'EOF'
apiVersion: 1

datasources:
  - name: Prometheus
    type: prometheus
    access: proxy
    url: http://prometheus:9090
    isDefault: true
    editable: true
EOF

    success "Monitoring dashboards created"
}

# Setup system service (if not using Docker)
setup_system_service() {
    log "Setting up system service for auto-scaler..."
    
    if command -v systemctl &> /dev/null; then
        cat > "$PROJECT_ROOT/scripts/ytempire-autoscaler.service" << EOF
[Unit]
Description=YTEmpire Auto-Scaler Service
After=network.target docker.service
Requires=docker.service

[Service]
Type=simple
User=$(whoami)
WorkingDirectory=$PROJECT_ROOT
Environment=PYTHONPATH=$PROJECT_ROOT
ExecStart=/usr/bin/python3 $PROJECT_ROOT/infrastructure/scaling/auto_scaler.py start --config $PROJECT_ROOT/infrastructure/scaling/scaling-config.json
Restart=always
RestartSec=30
StandardOutput=journal
StandardError=journal

[Install]
WantedBy=multi-user.target
EOF

        warning "Systemd service file created at $PROJECT_ROOT/scripts/ytempire-autoscaler.service"
        warning "To install: sudo cp $PROJECT_ROOT/scripts/ytempire-autoscaler.service /etc/systemd/system/"
        warning "Then run: sudo systemctl enable ytempire-autoscaler && sudo systemctl start ytempire-autoscaler"
    fi
}

# Create scaling test scripts
create_test_scripts() {
    log "Creating scaling test scripts..."
    
    # Load test script
    cat > "$PROJECT_ROOT/scripts/scaling/load-test.sh" << 'EOF'
#!/bin/bash
# Load testing script for auto-scaling

set -euo pipefail

BACKEND_URL="${1:-http://localhost:80}"
DURATION="${2:-300}"  # 5 minutes
CONCURRENT="${3:-50}"

echo "Starting load test against $BACKEND_URL"
echo "Duration: ${DURATION}s, Concurrent requests: $CONCURRENT"

# Install hey if not available
if ! command -v hey &> /dev/null; then
    echo "Installing hey load testing tool..."
    wget -O /tmp/hey https://hey-release.s3.us-east-2.amazonaws.com/hey_linux_amd64
    chmod +x /tmp/hey
    sudo mv /tmp/hey /usr/local/bin/hey
fi

# Run load test
hey -z "${DURATION}s" -c "$CONCURRENT" -q 10 "$BACKEND_URL/api/v1/health"

echo "Load test completed. Check Grafana dashboard for scaling events."
EOF

    # Scaling test script
    cat > "$PROJECT_ROOT/scripts/scaling/test-scaling.sh" << 'EOF'
#!/bin/bash
# Test auto-scaling functionality

set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"

echo "Testing YTEmpire Auto-Scaling System"

# Start scaling stack
echo "Starting scaling stack..."
cd "$PROJECT_ROOT"
docker-compose -f infrastructure/scaling/docker-compose.scaling.yml up -d

# Wait for services to be ready
echo "Waiting for services to be ready..."
sleep 30

# Check auto-scaler status
echo "Checking auto-scaler status..."
docker-compose -f infrastructure/scaling/docker-compose.scaling.yml exec -T auto_scaler \
  python /app/scaling/auto_scaler.py status

# Run load test
echo "Running load test to trigger scaling..."
./scripts/scaling/load-test.sh http://localhost:80 180 30

# Monitor scaling events
echo "Monitoring scaling events..."
docker-compose -f infrastructure/scaling/docker-compose.scaling.yml logs --tail=20 auto_scaler

echo "Scaling test completed. Check Grafana at http://localhost:3001 for detailed metrics."
EOF

    chmod +x "$PROJECT_ROOT/scripts/scaling/load-test.sh"
    chmod +x "$PROJECT_ROOT/scripts/scaling/test-scaling.sh"
    
    success "Test scripts created"
}

# Validate configuration
validate_configuration() {
    log "Validating scaling configuration..."
    
    cd "$PROJECT_ROOT"
    
    # Validate Docker Compose configuration
    if docker-compose -f infrastructure/scaling/docker-compose.scaling.yml config > /dev/null; then
        success "Docker Compose configuration is valid"
    else
        error "Docker Compose configuration validation failed"
    fi
    
    # Validate scaling configuration JSON
    if python3 -c "import json; json.load(open('infrastructure/scaling/scaling-config.json'))" 2>/dev/null; then
        success "Scaling configuration JSON is valid"
    else
        error "Scaling configuration JSON validation failed"
    fi
    
    # Check if required ports are available
    for port in 80 443 9090 3001 8404; do
        if ss -tuln | grep ":$port " >/dev/null; then
            warning "Port $port is already in use. This may cause conflicts."
        fi
    done
}

# Print setup instructions
print_instructions() {
    log "Auto-scaling setup completed successfully!"
    
    cat << EOF

${GREEN}YTEmpire Auto-Scaling Setup Complete!${NC}

${YELLOW}Next Steps:${NC}

1. Review and update environment variables:
   ${PROJECT_ROOT}/.env.scaling

2. Start the auto-scaling stack:
   cd ${PROJECT_ROOT}
   docker-compose -f infrastructure/scaling/docker-compose.scaling.yml up -d

3. Access monitoring dashboards:
   - Grafana: http://localhost:3001 (admin / check .env.scaling for password)
   - Prometheus: http://localhost:9090
   - HAProxy Stats: http://localhost:8404/haproxy-stats

4. Test auto-scaling:
   ./scripts/scaling/test-scaling.sh

5. Run load tests:
   ./scripts/scaling/load-test.sh http://localhost:80 300 50

6. Monitor scaling events:
   docker-compose -f infrastructure/scaling/docker-compose.scaling.yml logs -f auto_scaler

${YELLOW}Configuration Files:${NC}
- Environment: ${PROJECT_ROOT}/.env.scaling
- Scaling Config: ${PROJECT_ROOT}/infrastructure/scaling/scaling-config.json
- Docker Compose: ${PROJECT_ROOT}/infrastructure/scaling/docker-compose.scaling.yml
- HAProxy Config: ${PROJECT_ROOT}/infrastructure/scaling/haproxy.cfg

${YELLOW}Scaling Thresholds (configurable):${NC}
- CPU Scale Up: 80% (scale down: 30%)
- Memory Scale Up: 85% (scale down: 50%)
- Queue Scale Up: 50 items (scale down: 10 items)
- Response Time Scale Up: 2s (scale down: 500ms)

${YELLOW}Instance Limits:${NC}
- Backend: 1-10 instances (default: 2)
- Celery Workers: 1-20 instances (default: 3)
- Redis: 1-3 instances (default: 1)

${YELLOW}Service URLs:${NC}
- Application: http://localhost (load balanced)
- API: http://localhost/api/v1/
- WebSocket: ws://localhost/ws/
- Metrics: http://localhost:8001/metrics

${GREEN}Auto-scaling is now configured and ready to use!${NC}

EOF
}

# Main execution
main() {
    log "Starting YTEmpire Auto-Scaling Setup"
    
    # Create necessary directories
    mkdir -p "$PROJECT_ROOT/scripts/scaling"
    mkdir -p "$PROJECT_ROOT/infrastructure/scaling/grafana/dashboards"
    mkdir -p "$PROJECT_ROOT/infrastructure/scaling/grafana/provisioning/datasources"
    
    check_prerequisites
    setup_environment
    create_scaling_config
    setup_monitoring_dashboards
    setup_system_service
    create_test_scripts
    validate_configuration
    
    print_instructions
    
    success "Auto-scaling setup completed successfully!"
}

# Run main function with all arguments
main "$@"