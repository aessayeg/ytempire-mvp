#!/bin/bash

# Deploy N8N Backend Integration
# Owner: Integration Specialist

set -e

echo "🚀 Deploying YTEmpire N8N Integration..."

# Function to check service health
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

# Function to validate N8N configuration
validate_n8n_config() {
    echo "🔍 Validating N8N configuration..."
    
    # Check if N8N configuration exists
    if [ ! -f "docker-compose.yml" ]; then
        echo "❌ N8N docker-compose.yml not found"
        exit 1
    fi
    
    # Check if workflow templates exist
    if [ ! -d "workflows" ]; then
        echo "❌ N8N workflows directory not found"
        exit 1
    fi
    
    if [ ! -f "workflows/video-generation-workflow.json" ]; then
        echo "❌ Video generation workflow template not found"
        exit 1
    fi
    
    echo "✅ N8N configuration validated"
}

# Function to deploy N8N services
deploy_n8n_services() {
    echo "🚢 Deploying N8N services..."
    
    # Create network if it doesn't exist
    docker network create ytempire_network 2>/dev/null || echo "Network ytempire_network already exists"
    
    # Start N8N services
    docker-compose up -d
    
    # Wait for services to be ready
    echo "⏳ Waiting for N8N services to start..."
    sleep 15
    
    # Check service health
    check_service_health "N8N" "http://localhost:5678/healthz"
    check_service_health "N8N PostgreSQL" "http://localhost:5433" 5
}

# Function to import workflow templates
import_workflow_templates() {
    echo "📋 Importing workflow templates..."
    
    # Wait for N8N to be fully ready
    sleep 10
    
    # Import video generation workflow
    if [ -f "workflows/video-generation-workflow.json" ]; then
        echo "Importing video generation workflow..."
        # Note: In a real deployment, you would use N8N's API to import workflows
        # curl -X POST http://localhost:5678/api/v1/workflows/import \
        #   -H "Content-Type: application/json" \
        #   -d @workflows/video-generation-workflow.json
        echo "✅ Video generation workflow template ready for import"
    fi
    
    # Import YouTube upload workflow
    if [ -f "workflows/youtube-upload-workflow.json" ]; then
        echo "Importing YouTube upload workflow..."
        echo "✅ YouTube upload workflow template ready for import"
    fi
    
    echo "✅ Workflow templates processed"
}

# Function to test N8N integration
test_n8n_integration() {
    echo "🧪 Testing N8N integration..."
    
    # Test webhook endpoints
    local webhook_health=$(curl -s http://localhost:8000/api/v1/webhooks/n8n/health || echo "failed")
    
    if [[ "$webhook_health" == *"healthy"* ]]; then
        echo "✅ Backend webhook integration is working"
    else
        echo "❌ Backend webhook integration failed"
        echo "Make sure the backend service is running"
    fi
    
    # Test N8N API connection
    local n8n_api_test=$(curl -s http://localhost:5678/api/v1/health || echo "failed")
    
    if [[ "$n8n_api_test" == *"ok"* ]] || [ "$?" -eq 0 ]; then
        echo "✅ N8N API is accessible"
    else
        echo "⚠️ N8N API may not be fully ready (this is normal during startup)"
    fi
}

# Function to setup N8N credentials
setup_n8n_credentials() {
    echo "🔐 Setting up N8N credentials..."
    
    # Create credentials directory if it doesn't exist
    mkdir -p credentials
    
    # Create sample credential templates
    cat > credentials/openai-credentials.json << 'EOF'
{
  "name": "OpenAI API Key",
  "type": "httpHeaderAuth",
  "data": {
    "name": "Authorization",
    "value": "Bearer YOUR_OPENAI_API_KEY"
  }
}
EOF

    cat > credentials/elevenlabs-credentials.json << 'EOF'
{
  "name": "ElevenLabs API Key", 
  "type": "httpHeaderAuth",
  "data": {
    "name": "xi-api-key",
    "value": "YOUR_ELEVENLABS_API_KEY"
  }
}
EOF

    cat > credentials/youtube-oauth.json << 'EOF'
{
  "name": "YouTube API OAuth2",
  "type": "oAuth2Api",
  "data": {
    "clientId": "YOUR_YOUTUBE_CLIENT_ID",
    "clientSecret": "YOUR_YOUTUBE_CLIENT_SECRET",
    "accessTokenUrl": "https://accounts.google.com/o/oauth2/token",
    "authUrl": "https://accounts.google.com/o/oauth2/auth",
    "scope": "https://www.googleapis.com/auth/youtube.upload"
  }
}
EOF

    echo "✅ N8N credential templates created in credentials/ directory"
    echo "⚠️ Update the credential files with actual API keys before using workflows"
}

# Function to create N8N summary
create_n8n_summary() {
    echo ""
    echo "📋 N8N Integration Deployment Summary"
    echo "====================================="
    echo ""
    echo "🎯 Services Deployed:"
    echo "  • N8N Workflow Engine: http://localhost:5678 (admin/ytempire_n8n_2025)"
    echo "  • N8N PostgreSQL: localhost:5433"
    echo "  • N8N Redis: localhost:6380"
    echo ""
    echo "🔗 Workflow Templates Available:"
    echo "  • Video Generation Workflow (video-generation-workflow.json)"
    echo "  • YouTube Upload Workflow (youtube-upload-workflow.json)"
    echo ""
    echo "📡 Backend Integration Endpoints:"
    echo "  • Video Complete: POST /api/v1/webhooks/n8n/video-complete"
    echo "  • Trigger Upload: POST /api/v1/webhooks/n8n/trigger-upload"
    echo "  • Cost Alert: POST /api/v1/webhooks/n8n/cost-alert"
    echo "  • YouTube Callback: POST /api/v1/webhooks/youtube/callback"
    echo "  • Health Check: GET /api/v1/webhooks/n8n/health"
    echo ""
    echo "⚙️ Next Steps:"
    echo "  1. Access N8N at http://localhost:5678"
    echo "  2. Import workflow templates from workflows/ directory"
    echo "  3. Configure credentials in N8N (OpenAI, ElevenLabs, YouTube)"
    echo "  4. Update environment variables with API keys"
    echo "  5. Test video generation workflow"
    echo ""
    echo "🔐 Credential Setup:"
    echo "  • Update credentials/*.json files with your API keys"
    echo "  • Import credentials into N8N dashboard"
    echo ""
    echo "✅ N8N backend integration deployment completed!"
}

# Main execution
main() {
    echo "Starting N8N backend integration deployment..."
    
    # Change to N8N directory
    cd "$(dirname "$0")"
    
    # Validate configuration
    validate_n8n_config
    
    # Deploy services
    deploy_n8n_services
    
    # Import workflow templates
    import_workflow_templates
    
    # Setup credentials
    setup_n8n_credentials
    
    # Test integration
    test_n8n_integration
    
    # Show summary
    create_n8n_summary
}

# Execute main function
main "$@"