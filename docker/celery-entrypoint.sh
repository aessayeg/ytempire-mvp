#!/bin/bash
# Celery Worker Entrypoint Script
# Owner: DevOps Engineer

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

echo -e "${BLUE}YTEmpire Celery Worker Startup${NC}"
echo "================================"

# Environment validation
echo -e "${YELLOW}Validating environment...${NC}"

# Check required environment variables
REQUIRED_VARS=(
    "DATABASE_URL"
    "REDIS_URL" 
    "SECRET_KEY"
)

for var in "${REQUIRED_VARS[@]}"; do
    if [[ -z "${!var}" ]]; then
        echo -e "${RED}Error: Required environment variable $var is not set${NC}"
        exit 1
    fi
    echo -e "${GREEN}✓ $var is set${NC}"
done

# Wait for services to be ready
echo -e "${YELLOW}Waiting for services...${NC}"

# Wait for Redis
echo "Waiting for Redis..."
while ! redis-cli -u "$REDIS_URL" ping > /dev/null 2>&1; do
    sleep 1
done
echo -e "${GREEN}✓ Redis is ready${NC}"

# Wait for PostgreSQL
echo "Waiting for PostgreSQL..."
python -c "
import os
import time
import psycopg2
from urllib.parse import urlparse

url = urlparse(os.environ['DATABASE_URL'])
max_attempts = 30
attempt = 0

while attempt < max_attempts:
    try:
        conn = psycopg2.connect(
            host=url.hostname,
            port=url.port or 5432,
            user=url.username,
            password=url.password,
            database=url.path.lstrip('/')
        )
        conn.close()
        print('✓ PostgreSQL is ready')
        break
    except psycopg2.OperationalError:
        attempt += 1
        time.sleep(1)
        if attempt >= max_attempts:
            print('✗ Failed to connect to PostgreSQL')
            exit(1)
"

# Set default values for Celery configuration
export C_FORCE_ROOT=${C_FORCE_ROOT:-1}
export CELERY_WORKER_CONCURRENCY=${CELERY_WORKER_CONCURRENCY:-4}
export CELERY_WORKER_PREFETCH_MULTIPLIER=${CELERY_WORKER_PREFETCH_MULTIPLIER:-1}
export CELERY_WORKER_MAX_TASKS_PER_CHILD=${CELERY_WORKER_MAX_TASKS_PER_CHILD:-1000}
export CELERY_WORKER_LOG_LEVEL=${CELERY_WORKER_LOG_LEVEL:-info}

# Determine worker type and queues
WORKER_TYPE=${1:-default}
CELERY_QUEUES=""

case $WORKER_TYPE in
    "video")
        CELERY_QUEUES="video_generation,video_processing"
        echo -e "${BLUE}Starting Video Processing Worker${NC}"
        ;;
    "analytics")
        CELERY_QUEUES="analytics,reporting"
        echo -e "${BLUE}Starting Analytics Worker${NC}"
        ;;
    "ai")
        CELERY_QUEUES="ai_tasks,script_generation,voice_synthesis"
        echo -e "${BLUE}Starting AI Tasks Worker${NC}"
        ;;
    "low_priority")
        CELERY_QUEUES="low_priority,cleanup,maintenance"
        echo -e "${BLUE}Starting Low Priority Worker${NC}"
        ;;
    "beat")
        echo -e "${BLUE}Starting Celery Beat Scheduler${NC}"
        exec celery -A app.core.celery_app beat \
            --loglevel=$CELERY_WORKER_LOG_LEVEL \
            --schedule=/tmp/celerybeat-schedule \
            --pidfile=/tmp/celerybeat.pid
        ;;
    "flower")
        echo -e "${BLUE}Starting Celery Flower Monitor${NC}"
        exec celery -A app.core.celery_app flower \
            --port=5555 \
            --broker=$REDIS_URL
        ;;
    *)
        # Default worker handles all queues
        CELERY_QUEUES="default,video_generation,video_processing,analytics,ai_tasks"
        echo -e "${BLUE}Starting Default Worker (All Queues)${NC}"
        ;;
esac

echo -e "${YELLOW}Worker Configuration:${NC}"
echo "Worker Type: $WORKER_TYPE"
echo "Queues: $CELERY_QUEUES"
echo "Concurrency: $CELERY_WORKER_CONCURRENCY"
echo "Prefetch Multiplier: $CELERY_WORKER_PREFETCH_MULTIPLIER"
echo "Max Tasks Per Child: $CELERY_WORKER_MAX_TASKS_PER_CHILD"
echo "Log Level: $CELERY_WORKER_LOG_LEVEL"

# Performance tuning based on worker type
case $WORKER_TYPE in
    "video"|"ai")
        # High memory/CPU tasks - lower concurrency
        export CELERY_WORKER_CONCURRENCY=2
        export CELERY_WORKER_MAX_MEMORY_PER_CHILD=500000  # 500MB
        echo -e "${YELLOW}Applied performance tuning for high-resource tasks${NC}"
        ;;
    "analytics")
        # Medium resource tasks
        export CELERY_WORKER_CONCURRENCY=3
        export CELERY_WORKER_MAX_MEMORY_PER_CHILD=300000  # 300MB
        echo -e "${YELLOW}Applied performance tuning for analytics tasks${NC}"
        ;;
esac

# Create necessary directories
mkdir -p /tmp/celery
mkdir -p /app/logs

# Set up signal handlers for graceful shutdown
trap 'echo -e "${YELLOW}Shutting down Celery worker...${NC}"; exit 0' SIGINT SIGTERM

# Health check function
health_check() {
    celery -A app.core.celery_app inspect ping --timeout=10 > /dev/null 2>&1
    return $?
}

# Start health check in background
(
    sleep 30  # Wait for worker to start
    while true; do
        if ! health_check; then
            echo -e "${RED}Health check failed! Worker may be unresponsive.${NC}"
            # In production, you might want to restart or alert here
        fi
        sleep 60
    done
) &

# Create Celery worker command
CELERY_CMD="celery -A app.core.celery_app worker"

# Add queue specification
if [[ -n "$CELERY_QUEUES" ]]; then
    CELERY_CMD="$CELERY_CMD --queues=$CELERY_QUEUES"
fi

# Add other options
CELERY_CMD="$CELERY_CMD \
    --loglevel=$CELERY_WORKER_LOG_LEVEL \
    --concurrency=$CELERY_WORKER_CONCURRENCY \
    --prefetch-multiplier=$CELERY_WORKER_PREFETCH_MULTIPLIER \
    --max-tasks-per-child=$CELERY_WORKER_MAX_TASKS_PER_CHILD \
    --logfile=/app/logs/celery-$WORKER_TYPE.log \
    --pidfile=/tmp/celery-$WORKER_TYPE.pid"

# Add memory limit if set
if [[ -n "$CELERY_WORKER_MAX_MEMORY_PER_CHILD" ]]; then
    CELERY_CMD="$CELERY_CMD --max-memory-per-child=$CELERY_WORKER_MAX_MEMORY_PER_CHILD"
fi

# Add worker name for identification
WORKER_NAME="worker-$WORKER_TYPE@%h"
CELERY_CMD="$CELERY_CMD -n $WORKER_NAME"

echo -e "${GREEN}Starting Celery worker with command:${NC}"
echo "$CELERY_CMD"
echo ""

# Execute Celery worker
exec $CELERY_CMD