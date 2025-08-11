#!/bin/bash

# YTEmpire Staging Deployment Script
# Automated deployment with safety checks and rollback support

set -e  # Exit on error

# ================================================================================
# Configuration
# ================================================================================

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Deployment settings
PROJECT_NAME="ytempire"
ENVIRONMENT="staging"
DEPLOYMENT_VERSION="${1:-latest}"
DRY_RUN="${2:-false}"

# Paths
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
ENV_FILE="${PROJECT_ROOT}/.env.staging"
DOCKER_COMPOSE_FILE="${PROJECT_ROOT}/docker-compose.staging.yml"
BACKUP_DIR="${PROJECT_ROOT}/backups/staging"
LOG_DIR="${PROJECT_ROOT}/logs/staging"
LOG_FILE="${LOG_DIR}/deployment_$(date +%Y%m%d_%H%M%S).log"

# Create directories
mkdir -p "$BACKUP_DIR" "$LOG_DIR"

# ================================================================================
# Functions
# ================================================================================

log_info() {
    local message="$1"
    echo -e "${BLUE}[$(date +'%Y-%m-%d %H:%M:%S')] INFO:${NC} $message" | tee -a "$LOG_FILE"
}

log_success() {
    local message="$1"
    echo -e "${GREEN}[$(date +'%Y-%m-%d %H:%M:%S')] SUCCESS:${NC} $message" | tee -a "$LOG_FILE"
}

log_warning() {
    local message="$1"
    echo -e "${YELLOW}[$(date +'%Y-%m-%d %H:%M:%S')] WARNING:${NC} $message" | tee -a "$LOG_FILE"
}

log_error() {
    local message="$1"
    echo -e "${RED}[$(date +'%Y-%m-%d %H:%M:%S')] ERROR:${NC} $message" | tee -a "$LOG_FILE"
}

# Check prerequisites
check_prerequisites() {
    log_info "Checking prerequisites..."
    
    # Check Docker
    if ! command -v docker &> /dev/null; then
        log_error "Docker is not installed"
        exit 1
    fi
    
    # Check Docker Compose
    if ! command -v docker-compose &> /dev/null; then
        log_error "Docker Compose is not installed"
        exit 1
    fi
    
    # Check environment file
    if [ ! -f "$ENV_FILE" ]; then
        log_warning "Environment file not found: $ENV_FILE"
        log_info "Creating from template..."
        cp "${PROJECT_ROOT}/.env.template" "$ENV_FILE"
        log_warning "Please update $ENV_FILE with staging values"
    fi
    
    # Check disk space
    AVAILABLE_SPACE=$(df -BG . | awk 'NR==2 {print $4}' | sed 's/G//')
    if [ "$AVAILABLE_SPACE" -lt 5 ]; then
        log_error "Insufficient disk space: ${AVAILABLE_SPACE}GB available (minimum 5GB required)"
        exit 1
    fi
    
    log_success "Prerequisites check passed"
}

# Create backup
create_backup() {
    log_info "Creating backup..."
    
    TIMESTAMP=$(date +%Y%m%d_%H%M%S)
    BACKUP_PATH="${BACKUP_DIR}/backup_${TIMESTAMP}"
    mkdir -p "$BACKUP_PATH"
    
    # Backup database
    if docker ps | grep -q "${PROJECT_NAME}_postgres_staging"; then
        log_info "Backing up database..."
        docker exec "${PROJECT_NAME}_postgres_staging" \
            pg_dump -U ytempire ytempire_staging > "${BACKUP_PATH}/database.sql" 2>/dev/null || {
            log_warning "Database backup failed (container might be starting)"
        }
    fi
    
    # Backup environment file
    cp "$ENV_FILE" "${BACKUP_PATH}/.env.backup"
    
    # Backup uploads if they exist
    if [ -d "${PROJECT_ROOT}/uploads_staging" ]; then
        tar czf "${BACKUP_PATH}/uploads.tar.gz" -C "${PROJECT_ROOT}" uploads_staging 2>/dev/null || true
    fi
    
    # Create backup manifest
    cat > "${BACKUP_PATH}/manifest.json" << EOF
{
    "timestamp": "$TIMESTAMP",
    "environment": "staging",
    "version": "$DEPLOYMENT_VERSION",
    "created_by": "$(whoami)",
    "hostname": "$(hostname)"
}
EOF
    
    log_success "Backup created at: $BACKUP_PATH"
    echo "$BACKUP_PATH" > "${BACKUP_DIR}/.last_backup"
}

# Pull Docker images
pull_images() {
    log_info "Pulling Docker images..."
    
    if [ "$DRY_RUN" == "true" ]; then
        log_info "DRY RUN: Would pull images with version $DEPLOYMENT_VERSION"
        return 0
    fi
    
    # Pull images specified in docker-compose
    docker-compose -f "$DOCKER_COMPOSE_FILE" pull || {
        log_error "Failed to pull Docker images"
        exit 1
    }
    
    log_success "Docker images pulled successfully"
}

# Run database migrations
run_migrations() {
    log_info "Running database migrations..."
    
    if [ "$DRY_RUN" == "true" ]; then
        log_info "DRY RUN: Would run database migrations"
        return 0
    fi
    
    # Start database first
    docker-compose -f "$DOCKER_COMPOSE_FILE" up -d postgres redis
    
    # Wait for database to be ready
    log_info "Waiting for database to be ready..."
    for i in {1..30}; do
        if docker-compose -f "$DOCKER_COMPOSE_FILE" exec -T postgres pg_isready -U ytempire &>/dev/null; then
            log_success "Database is ready"
            break
        fi
        if [ $i -eq 30 ]; then
            log_error "Database failed to start"
            exit 1
        fi
        sleep 2
    done
    
    # Run Alembic migrations
    docker-compose -f "$DOCKER_COMPOSE_FILE" run --rm backend \
        alembic upgrade head || {
        log_error "Database migration failed"
        exit 1
    }
    
    log_success "Database migrations completed"
}

# Deploy services
deploy_services() {
    log_info "Deploying services..."
    
    if [ "$DRY_RUN" == "true" ]; then
        log_info "DRY RUN: Would deploy services"
        docker-compose -f "$DOCKER_COMPOSE_FILE" config
        return 0
    fi
    
    # Deploy services in order
    SERVICES=(
        "postgres"
        "redis"
        "backend"
        "frontend"
        "celery_worker"
        "celery_beat"
        "flower"
        "nginx"
        "prometheus"
        "grafana"
    )
    
    for service in "${SERVICES[@]}"; do
        log_info "Starting $service..."
        docker-compose -f "$DOCKER_COMPOSE_FILE" up -d --no-deps "$service" || {
            log_error "Failed to start $service"
            rollback
            exit 1
        }
        
        # Brief pause between services
        sleep 2
    done
    
    log_success "All services deployed"
}

# Health checks
health_check() {
    log_info "Performing health checks..."
    
    local all_healthy=true
    
    # Check backend health
    log_info "Checking backend health..."
    for i in {1..30}; do
        if curl -f http://localhost:8001/api/v1/health &>/dev/null; then
            log_success "Backend is healthy"
            break
        fi
        if [ $i -eq 30 ]; then
            log_error "Backend health check failed"
            all_healthy=false
        fi
        sleep 5
    done
    
    # Check frontend health
    log_info "Checking frontend health..."
    for i in {1..30}; do
        if curl -f http://localhost:3001 &>/dev/null; then
            log_success "Frontend is healthy"
            break
        fi
        if [ $i -eq 30 ]; then
            log_error "Frontend health check failed"
            all_healthy=false
        fi
        sleep 5
    done
    
    # Check database
    if docker-compose -f "$DOCKER_COMPOSE_FILE" exec -T postgres pg_isready -U ytempire &>/dev/null; then
        log_success "PostgreSQL is healthy"
    else
        log_error "PostgreSQL health check failed"
        all_healthy=false
    fi
    
    # Check Redis
    if docker-compose -f "$DOCKER_COMPOSE_FILE" exec -T redis redis-cli ping &>/dev/null; then
        log_success "Redis is healthy"
    else
        log_error "Redis health check failed"
        all_healthy=false
    fi
    
    if [ "$all_healthy" = false ]; then
        log_error "Health checks failed"
        return 1
    fi
    
    log_success "All health checks passed"
    return 0
}

# Generate test data
generate_test_data() {
    log_info "Generating test data for staging..."
    
    if [ "$DRY_RUN" == "true" ]; then
        log_info "DRY RUN: Would generate test data"
        return 0
    fi
    
    # Run test data generator
    docker-compose -f "$DOCKER_COMPOSE_FILE" run --rm test_data_generator || {
        log_warning "Test data generation failed (may already exist)"
    }
    
    log_success "Test data generation completed"
}

# Rollback deployment
rollback() {
    log_warning "Initiating rollback..."
    
    # Get last backup path
    if [ -f "${BACKUP_DIR}/.last_backup" ]; then
        LAST_BACKUP=$(cat "${BACKUP_DIR}/.last_backup")
        
        if [ -d "$LAST_BACKUP" ]; then
            log_info "Restoring from backup: $LAST_BACKUP"
            
            # Restore database
            if [ -f "${LAST_BACKUP}/database.sql" ]; then
                docker-compose -f "$DOCKER_COMPOSE_FILE" exec -T postgres \
                    psql -U ytempire ytempire_staging < "${LAST_BACKUP}/database.sql" 2>/dev/null || {
                    log_warning "Database restore failed"
                }
            fi
            
            # Restore environment
            if [ -f "${LAST_BACKUP}/.env.backup" ]; then
                cp "${LAST_BACKUP}/.env.backup" "$ENV_FILE"
            fi
            
            # Restart services
            docker-compose -f "$DOCKER_COMPOSE_FILE" restart
            
            log_success "Rollback completed"
        else
            log_error "Backup directory not found: $LAST_BACKUP"
        fi
    else
        log_error "No backup reference found"
    fi
}

# Cleanup old resources
cleanup() {
    log_info "Cleaning up old resources..."
    
    # Remove unused Docker images
    docker image prune -f
    
    # Remove old backups (keep last 7 days)
    find "$BACKUP_DIR" -type d -name "backup_*" -mtime +7 -exec rm -rf {} + 2>/dev/null || true
    
    # Clean old logs (keep last 30 days)
    find "$LOG_DIR" -type f -name "*.log" -mtime +30 -delete 2>/dev/null || true
    
    log_success "Cleanup completed"
}

# Show deployment status
show_status() {
    echo ""
    echo "========================================"
    echo "Staging Deployment Status"
    echo "========================================"
    docker-compose -f "$DOCKER_COMPOSE_FILE" ps
    echo "========================================"
    echo ""
    echo "Access URLs:"
    echo "  - Frontend: http://localhost:3001"
    echo "  - Backend API: http://localhost:8001"
    echo "  - API Docs: http://localhost:8001/docs"
    echo "  - Flower (Celery): http://localhost:5556"
    echo "  - Grafana: http://localhost:3002"
    echo "  - Prometheus: http://localhost:9091"
    echo "  - Mailhog: http://localhost:8025"
    echo "========================================"
}

# ================================================================================
# Main Deployment Process
# ================================================================================

main() {
    log_info "Starting YTEmpire staging deployment..."
    log_info "Version: $DEPLOYMENT_VERSION"
    log_info "Dry Run: $DRY_RUN"
    
    # Change to project root
    cd "$PROJECT_ROOT"
    
    # Load environment variables
    if [ -f "$ENV_FILE" ]; then
        set -a
        source "$ENV_FILE"
        set +a
    fi
    
    # Step 1: Prerequisites check
    check_prerequisites
    
    # Step 2: Create backup
    if [ "$DRY_RUN" != "true" ]; then
        create_backup
    fi
    
    # Step 3: Pull images
    pull_images
    
    # Step 4: Run migrations
    run_migrations
    
    # Step 5: Deploy services
    deploy_services
    
    # Step 6: Health checks
    if health_check; then
        log_success "Deployment completed successfully!"
        
        # Step 7: Generate test data (first deployment only)
        if [ ! -f "${PROJECT_ROOT}/.staging_initialized" ]; then
            generate_test_data
            touch "${PROJECT_ROOT}/.staging_initialized"
        fi
        
        # Step 8: Cleanup
        cleanup
        
        # Show status
        show_status
        
        exit 0
    else
        log_error "Deployment failed!"
        
        if [ "$DRY_RUN" != "true" ]; then
            # Ask for rollback
            read -p "Do you want to rollback? (y/n): " -n 1 -r
            echo
            if [[ $REPLY =~ ^[Yy]$ ]]; then
                rollback
            fi
        fi
        
        exit 1
    fi
}

# ================================================================================
# Script Entry Point
# ================================================================================

case "${1:-deploy}" in
    deploy)
        main
        ;;
    rollback)
        rollback
        ;;
    status)
        show_status
        ;;
    cleanup)
        cleanup
        ;;
    health)
        health_check
        ;;
    backup)
        create_backup
        ;;
    *)
        echo "Usage: $0 {deploy|rollback|status|cleanup|health|backup} [version] [dry-run]"
        echo ""
        echo "Commands:"
        echo "  deploy [version] [dry-run] - Deploy to staging environment"
        echo "  rollback                   - Rollback to previous deployment"
        echo "  status                     - Show deployment status"
        echo "  cleanup                    - Clean up old resources"
        echo "  health                     - Run health checks"
        echo "  backup                     - Create backup only"
        echo ""
        echo "Examples:"
        echo "  $0 deploy latest          - Deploy latest version"
        echo "  $0 deploy v1.2.3 true     - Dry run for version v1.2.3"
        echo "  $0 rollback               - Rollback to previous version"
        exit 1
        ;;
esac