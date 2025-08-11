#!/bin/bash

# YTEmpire Production Deployment Script
# Automated deployment with zero-downtime and rollback support

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
DEPLOYMENT_ENV="${1:-production}"
DEPLOYMENT_VERSION="${2:-latest}"
BACKUP_ENABLED=true
HEALTH_CHECK_RETRIES=5
HEALTH_CHECK_INTERVAL=10

# Paths
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
ENV_FILE="${PROJECT_ROOT}/.env.${DEPLOYMENT_ENV}"
DOCKER_COMPOSE_FILE="${PROJECT_ROOT}/docker-compose.yml"
DOCKER_COMPOSE_PROD="${PROJECT_ROOT}/docker-compose.production.yml"
BACKUP_DIR="${PROJECT_ROOT}/backups"
LOG_DIR="${PROJECT_ROOT}/logs"

# ================================================================================
# Functions
# ================================================================================

log_info() {
    echo -e "${BLUE}[$(date +'%Y-%m-%d %H:%M:%S')] INFO:${NC} $1"
}

log_success() {
    echo -e "${GREEN}[$(date +'%Y-%m-%d %H:%M:%S')] SUCCESS:${NC} $1"
}

log_warning() {
    echo -e "${YELLOW}[$(date +'%Y-%m-%d %H:%M:%S')] WARNING:${NC} $1"
}

log_error() {
    echo -e "${RED}[$(date +'%Y-%m-%d %H:%M:%S')] ERROR:${NC} $1"
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
        log_error "Environment file not found: $ENV_FILE"
        log_info "Please create $ENV_FILE from .env.template"
        exit 1
    fi
    
    # Check disk space
    AVAILABLE_SPACE=$(df -BG . | awk 'NR==2 {print $4}' | sed 's/G//')
    if [ "$AVAILABLE_SPACE" -lt 10 ]; then
        log_warning "Low disk space: ${AVAILABLE_SPACE}GB available"
    fi
    
    log_success "Prerequisites check passed"
}

# Backup current deployment
backup_deployment() {
    if [ "$BACKUP_ENABLED" = true ]; then
        log_info "Creating backup..."
        
        TIMESTAMP=$(date +%Y%m%d_%H%M%S)
        BACKUP_PATH="${BACKUP_DIR}/deployment_${TIMESTAMP}"
        mkdir -p "$BACKUP_PATH"
        
        # Backup database
        log_info "Backing up database..."
        docker-compose -f "$DOCKER_COMPOSE_PROD" exec -T postgres \
            pg_dump -U ytempire ytempire_db > "${BACKUP_PATH}/database.sql" 2>/dev/null || true
        
        # Backup environment file
        cp "$ENV_FILE" "${BACKUP_PATH}/.env.backup"
        
        # Backup volumes
        log_info "Backing up Docker volumes..."
        docker run --rm \
            -v ytempire_postgres_data:/source:ro \
            -v "${BACKUP_PATH}":/backup \
            alpine tar czf /backup/postgres_data.tar.gz -C /source . 2>/dev/null || true
        
        log_success "Backup created at: $BACKUP_PATH"
    else
        log_warning "Backup is disabled"
    fi
}

# Build Docker images
build_images() {
    log_info "Building Docker images..."
    
    # Set build target
    export BUILD_TARGET=production
    
    # Build backend
    log_info "Building backend image..."
    docker build \
        --target production \
        --cache-from ${PROJECT_NAME}_backend:latest \
        -t ${PROJECT_NAME}_backend:${DEPLOYMENT_VERSION} \
        -t ${PROJECT_NAME}_backend:latest \
        ./backend
    
    # Build frontend
    log_info "Building frontend image..."
    docker build \
        --target production \
        --build-arg VITE_API_URL=https://api.ytempire.com \
        --cache-from ${PROJECT_NAME}_frontend:latest \
        -t ${PROJECT_NAME}_frontend:${DEPLOYMENT_VERSION} \
        -t ${PROJECT_NAME}_frontend:latest \
        ./frontend
    
    # Build ML pipeline
    if [ -d "./ml-pipeline" ]; then
        log_info "Building ML pipeline image..."
        docker build \
            --cache-from ${PROJECT_NAME}_ml_pipeline:latest \
            -t ${PROJECT_NAME}_ml_pipeline:${DEPLOYMENT_VERSION} \
            -t ${PROJECT_NAME}_ml_pipeline:latest \
            ./ml-pipeline
    fi
    
    log_success "Docker images built successfully"
}

# Run database migrations
run_migrations() {
    log_info "Running database migrations..."
    
    # Start only database service
    docker-compose -f "$DOCKER_COMPOSE_PROD" up -d postgres redis
    sleep 10  # Wait for database to be ready
    
    # Run Alembic migrations
    docker-compose -f "$DOCKER_COMPOSE_PROD" run --rm backend \
        alembic upgrade head
    
    log_success "Database migrations completed"
}

# Deploy services with zero-downtime
deploy_services() {
    log_info "Deploying services..."
    
    # Deploy in order of dependencies
    SERVICES=(
        "postgres"
        "redis"
        "backend"
        "celery_worker"
        "celery_beat"
        "ml_pipeline"
        "frontend"
        "nginx"
        "prometheus"
        "grafana"
        "n8n"
    )
    
    for service in "${SERVICES[@]}"; do
        log_info "Deploying $service..."
        
        # Scale down old container (for stateless services)
        if [[ "$service" == "backend" || "$service" == "frontend" || "$service" == "celery_worker" ]]; then
            # Rolling update for zero-downtime
            docker-compose -f "$DOCKER_COMPOSE_PROD" up -d --no-deps --scale ${service}=2 ${service}
            sleep 5
            
            # Health check
            if ! health_check "$service"; then
                log_error "Health check failed for $service"
                rollback_deployment
                exit 1
            fi
            
            # Remove old container
            docker-compose -f "$DOCKER_COMPOSE_PROD" rm -f -s ${service}_old 2>/dev/null || true
        else
            # Regular deployment for stateful services
            docker-compose -f "$DOCKER_COMPOSE_PROD" up -d --no-deps ${service}
        fi
    done
    
    log_success "All services deployed"
}

# Health check function
health_check() {
    local service=$1
    local retries=$HEALTH_CHECK_RETRIES
    
    log_info "Performing health check for $service..."
    
    while [ $retries -gt 0 ]; do
        case "$service" in
            backend)
                if curl -f http://localhost:8000/health &>/dev/null; then
                    log_success "Backend is healthy"
                    return 0
                fi
                ;;
            frontend)
                if curl -f http://localhost:3000 &>/dev/null; then
                    log_success "Frontend is healthy"
                    return 0
                fi
                ;;
            postgres)
                if docker-compose -f "$DOCKER_COMPOSE_PROD" exec -T postgres pg_isready &>/dev/null; then
                    log_success "PostgreSQL is healthy"
                    return 0
                fi
                ;;
            redis)
                if docker-compose -f "$DOCKER_COMPOSE_PROD" exec -T redis redis-cli ping &>/dev/null; then
                    log_success "Redis is healthy"
                    return 0
                fi
                ;;
            *)
                # Generic container health check
                if docker ps | grep -q "${PROJECT_NAME}_${service}"; then
                    log_success "$service is running"
                    return 0
                fi
                ;;
        esac
        
        retries=$((retries - 1))
        if [ $retries -gt 0 ]; then
            log_warning "Health check failed for $service, retrying in ${HEALTH_CHECK_INTERVAL}s... ($retries retries left)"
            sleep $HEALTH_CHECK_INTERVAL
        fi
    done
    
    log_error "Health check failed for $service after $HEALTH_CHECK_RETRIES attempts"
    return 1
}

# Verify deployment
verify_deployment() {
    log_info "Verifying deployment..."
    
    # Check all services are running
    EXPECTED_SERVICES=(
        "${PROJECT_NAME}_postgres"
        "${PROJECT_NAME}_redis"
        "${PROJECT_NAME}_backend"
        "${PROJECT_NAME}_frontend"
        "${PROJECT_NAME}_celery_worker"
        "${PROJECT_NAME}_celery_beat"
    )
    
    for service in "${EXPECTED_SERVICES[@]}"; do
        if ! docker ps | grep -q "$service"; then
            log_error "$service is not running"
            return 1
        fi
    done
    
    # Run smoke tests
    log_info "Running smoke tests..."
    
    # Test API endpoint
    if ! curl -f http://localhost:8000/api/v1/health &>/dev/null; then
        log_error "API health check failed"
        return 1
    fi
    
    # Test frontend
    if ! curl -f http://localhost:3000 &>/dev/null; then
        log_error "Frontend check failed"
        return 1
    fi
    
    # Test database connection
    if ! docker-compose -f "$DOCKER_COMPOSE_PROD" exec -T backend \
        python -c "import psycopg2; psycopg2.connect('${DATABASE_URL}')" &>/dev/null; then
        log_error "Database connection test failed"
        return 1
    fi
    
    log_success "Deployment verification passed"
    return 0
}

# Rollback deployment
rollback_deployment() {
    log_warning "Rolling back deployment..."
    
    # Find latest backup
    if [ -d "$BACKUP_DIR" ]; then
        LATEST_BACKUP=$(ls -t "$BACKUP_DIR" | head -1)
        if [ -n "$LATEST_BACKUP" ]; then
            log_info "Restoring from backup: $LATEST_BACKUP"
            
            # Restore database
            if [ -f "${BACKUP_DIR}/${LATEST_BACKUP}/database.sql" ]; then
                docker-compose -f "$DOCKER_COMPOSE_PROD" exec -T postgres \
                    psql -U ytempire ytempire_db < "${BACKUP_DIR}/${LATEST_BACKUP}/database.sql"
            fi
            
            # Restore environment
            if [ -f "${BACKUP_DIR}/${LATEST_BACKUP}/.env.backup" ]; then
                cp "${BACKUP_DIR}/${LATEST_BACKUP}/.env.backup" "$ENV_FILE"
            fi
            
            # Restart services with previous version
            docker-compose -f "$DOCKER_COMPOSE_PROD" down
            docker-compose -f "$DOCKER_COMPOSE_PROD" up -d
            
            log_success "Rollback completed"
        else
            log_error "No backup found for rollback"
        fi
    else
        log_error "Backup directory not found"
    fi
}

# Cleanup old resources
cleanup() {
    log_info "Cleaning up old resources..."
    
    # Remove unused Docker images
    docker image prune -f
    
    # Remove old backups (keep last 7 days)
    if [ -d "$BACKUP_DIR" ]; then
        find "$BACKUP_DIR" -type d -mtime +7 -exec rm -rf {} + 2>/dev/null || true
    fi
    
    # Clean old logs
    if [ -d "$LOG_DIR" ]; then
        find "$LOG_DIR" -type f -name "*.log" -mtime +30 -delete 2>/dev/null || true
    fi
    
    log_success "Cleanup completed"
}

# Send deployment notification
send_notification() {
    local status=$1
    local message=$2
    
    # Slack notification (if configured)
    if [ -n "$SLACK_WEBHOOK_URL" ]; then
        curl -X POST -H 'Content-type: application/json' \
            --data "{\"text\":\"Deployment ${status}: ${message}\"}" \
            "$SLACK_WEBHOOK_URL" 2>/dev/null || true
    fi
    
    # Email notification (if configured)
    if [ -n "$NOTIFICATION_EMAIL" ]; then
        echo "$message" | mail -s "YTEmpire Deployment ${status}" "$NOTIFICATION_EMAIL" 2>/dev/null || true
    fi
}

# ================================================================================
# Main Deployment Process
# ================================================================================

main() {
    log_info "Starting YTEmpire production deployment..."
    log_info "Environment: $DEPLOYMENT_ENV"
    log_info "Version: $DEPLOYMENT_VERSION"
    
    # Change to project root
    cd "$PROJECT_ROOT"
    
    # Load environment variables
    set -a
    source "$ENV_FILE"
    set +a
    
    # Step 1: Prerequisites check
    check_prerequisites
    
    # Step 2: Create backup
    backup_deployment
    
    # Step 3: Build images
    build_images
    
    # Step 4: Run migrations
    run_migrations
    
    # Step 5: Deploy services
    deploy_services
    
    # Step 6: Verify deployment
    if verify_deployment; then
        log_success "Deployment completed successfully!"
        send_notification "SUCCESS" "YTEmpire deployed to $DEPLOYMENT_ENV (version: $DEPLOYMENT_VERSION)"
        
        # Step 7: Cleanup
        cleanup
        
        # Print summary
        echo ""
        echo "======================================"
        echo "Deployment Summary:"
        echo "======================================"
        echo "Environment: $DEPLOYMENT_ENV"
        echo "Version: $DEPLOYMENT_VERSION"
        echo "Backend URL: ${BACKEND_URL:-http://localhost:8000}"
        echo "Frontend URL: ${FRONTEND_URL:-http://localhost:3000}"
        echo "======================================"
        
        # Show service status
        docker-compose -f "$DOCKER_COMPOSE_PROD" ps
        
        exit 0
    else
        log_error "Deployment verification failed!"
        send_notification "FAILED" "YTEmpire deployment failed for $DEPLOYMENT_ENV"
        
        # Ask for rollback
        read -p "Do you want to rollback? (y/n): " -n 1 -r
        echo
        if [[ $REPLY =~ ^[Yy]$ ]]; then
            rollback_deployment
        fi
        
        exit 1
    fi
}

# ================================================================================
# Script Entry Point
# ================================================================================

# Handle script arguments
case "${1:-}" in
    production|staging|development)
        main
        ;;
    rollback)
        rollback_deployment
        ;;
    verify)
        verify_deployment
        ;;
    cleanup)
        cleanup
        ;;
    *)
        echo "Usage: $0 {production|staging|development|rollback|verify|cleanup} [version]"
        echo ""
        echo "Commands:"
        echo "  production [version]  - Deploy to production environment"
        echo "  staging [version]     - Deploy to staging environment"
        echo "  development [version] - Deploy to development environment"
        echo "  rollback             - Rollback to previous deployment"
        echo "  verify               - Verify current deployment"
        echo "  cleanup              - Clean up old resources"
        exit 1
        ;;
esac