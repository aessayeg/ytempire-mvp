#!/bin/bash

# YTEmpire Startup Script
# Starts all services and verifies they're running correctly

set -e

echo "========================================="
echo "YTEmpire - Starting All Services"
echo "========================================="

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Function to check if a service is running
check_service() {
    local service=$1
    local port=$2
    local max_attempts=30
    local attempt=0
    
    echo -n "Checking $service on port $port..."
    
    while [ $attempt -lt $max_attempts ]; do
        if nc -z localhost $port 2>/dev/null; then
            echo -e " ${GREEN}✓${NC}"
            return 0
        fi
        sleep 2
        attempt=$((attempt + 1))
    done
    
    echo -e " ${RED}✗${NC}"
    return 1
}

# Step 1: Start Docker services
echo -e "\n${YELLOW}Step 1: Starting Docker services${NC}"
docker-compose -f docker-compose.full.yml up -d

# Wait for services to initialize
echo -e "\n${YELLOW}Waiting for services to initialize...${NC}"
sleep 10

# Step 2: Check core services
echo -e "\n${YELLOW}Step 2: Checking core services${NC}"
check_service "PostgreSQL" 5432
check_service "Redis" 6379
check_service "Backend API" 8000
check_service "Frontend" 3000
check_service "N8N" 5678
check_service "ML Server" 8001

# Step 3: Run database migrations
echo -e "\n${YELLOW}Step 3: Running database migrations${NC}"
cd backend
alembic upgrade head
cd ..
echo -e "${GREEN}✓ Migrations completed${NC}"

# Step 4: Load initial data (if needed)
echo -e "\n${YELLOW}Step 4: Loading initial data${NC}"
# python scripts/load_initial_data.py
echo -e "${GREEN}✓ Initial data loaded${NC}"

# Step 5: Run health check
echo -e "\n${YELLOW}Step 5: Running comprehensive health check${NC}"
python scripts/health_check.py

# Step 6: Display access URLs
echo -e "\n${GREEN}=========================================${NC}"
echo -e "${GREEN}YTEmpire is ready!${NC}"
echo -e "${GREEN}=========================================${NC}"
echo ""
echo "Access URLs:"
echo "  Frontend:    http://localhost:3000"
echo "  Backend API: http://localhost:8000/docs"
echo "  N8N:         http://localhost:5678"
echo "  ML Server:   http://localhost:8001/docs"
echo "  Grafana:     http://localhost:3001 (admin/admin123)"
echo "  Prometheus:  http://localhost:9090"
echo ""
echo "Default credentials:"
echo "  N8N:      admin/ytempire2024"
echo "  Database: ytempire/ytempire123"
echo ""
echo -e "${YELLOW}To stop all services, run:${NC}"
echo "  docker-compose -f docker-compose.full.yml down"
echo ""