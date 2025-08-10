#!/bin/bash

# Deploy Monitoring Stack
# Owner: Platform Ops Lead

set -e

echo "🚀 Deploying YTEmpire Monitoring Stack..."

# Function to check if service is healthy
check_service_health() {
    local service_name=$1
    local url=$2
    local max_retries=${3:-30}
    local retry_count=0
    
    echo "Checking health of $service_name..."
    
    while [ $retry_count -lt $max_retries ]; do
        if curl -f "$url" >/dev/null 2>&1; then
            echo "✅ $service_name is healthy"
            return 0
        fi
        
        retry_count=$((retry_count + 1))
        echo "⏳ Waiting for $service_name to be ready... ($retry_count/$max_retries)"
        sleep 5
    done
    
    echo "❌ $service_name failed to become healthy"
    return 1
}

# Function to validate monitoring configuration
validate_monitoring_config() {
    echo "🔍 Validating monitoring configuration..."
    
    # Check if Prometheus config exists
    if [ ! -f "infrastructure/prometheus/prometheus.yml" ]; then
        echo "❌ Prometheus configuration not found"
        exit 1
    fi
    
    # Check if Grafana dashboards exist
    if [ ! -d "infrastructure/grafana/dashboards" ]; then
        echo "❌ Grafana dashboards directory not found"
        exit 1
    fi
    
    echo "✅ Monitoring configuration validated"
}

# Function to deploy monitoring services
deploy_monitoring_services() {
    echo "🚢 Deploying monitoring services..."
    
    # Start monitoring services
    docker-compose up -d prometheus grafana
    
    # Wait for services to be ready
    echo "⏳ Waiting for services to start..."
    sleep 10
    
    # Check service health
    check_service_health "Prometheus" "http://localhost:9090/-/healthy"
    check_service_health "Grafana" "http://localhost:3001/api/health"
}

# Function to configure Grafana
configure_grafana() {
    echo "⚙️ Configuring Grafana..."
    
    # Wait for Grafana to be fully ready
    sleep 15
    
    # Check if datasource is configured
    local grafana_health=$(curl -u admin:ytempire_grafana http://localhost:3001/api/health 2>/dev/null || echo "failed")
    
    if [[ "$grafana_health" == *"ok"* ]]; then
        echo "✅ Grafana is configured and ready"
    else
        echo "❌ Grafana configuration failed"
        return 1
    fi
}

# Function to test metrics collection
test_metrics_collection() {
    echo "📊 Testing metrics collection..."
    
    # Test Prometheus metrics endpoint
    if curl -f http://localhost:8000/metrics >/dev/null 2>&1; then
        echo "✅ Backend metrics endpoint is accessible"
    else
        echo "❌ Backend metrics endpoint is not accessible"
        echo "Make sure the backend service is running"
    fi
    
    # Test if Prometheus can scrape backend metrics
    sleep 10  # Allow time for scraping
    local metrics_available=$(curl -s "http://localhost:9090/api/v1/query?query=up{job='backend-api'}" | grep -o '"value":\[.*,"1"\]' || echo "")
    
    if [ -n "$metrics_available" ]; then
        echo "✅ Prometheus is successfully scraping backend metrics"
    else
        echo "⚠️ Prometheus is not yet scraping backend metrics (this is normal if backend isn't running)"
    fi
}

# Function to create monitoring summary
create_monitoring_summary() {
    echo ""
    echo "📋 Monitoring Stack Deployment Summary"
    echo "======================================"
    echo ""
    echo "🎯 Services Deployed:"
    echo "  • Prometheus: http://localhost:9090"
    echo "  • Grafana: http://localhost:3001 (admin/ytempire_grafana)"
    echo ""
    echo "📊 Available Dashboards:"
    echo "  • YTEmpire Backend Dashboard"
    echo ""
    echo "⚠️ Alert Rules Configured:"
    echo "  • High Error Rate"
    echo "  • High Response Time"
    echo "  • Service Down"
    echo "  • Database/Redis Down"
    echo ""
    echo "🔗 Key URLs:"
    echo "  • Prometheus: http://localhost:9090"
    echo "  • Grafana: http://localhost:3001"
    echo "  • Backend Metrics: http://localhost:8000/metrics"
    echo "  • Backend Health: http://localhost:8000/health"
    echo ""
    echo "✅ Monitoring stack deployment completed!"
}

# Main execution
main() {
    echo "Starting monitoring stack deployment..."
    
    # Change to project root
    cd "$(dirname "$0")/../.."
    
    # Validate configuration
    validate_monitoring_config
    
    # Deploy services
    deploy_monitoring_services
    
    # Configure Grafana
    configure_grafana
    
    # Test metrics collection
    test_metrics_collection
    
    # Show summary
    create_monitoring_summary
}

# Execute main function
main "$@"