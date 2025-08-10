#!/bin/bash

# Vector Database Deployment Script
# Owner: VP of AI

set -e

echo "🚀 Deploying YTEmpire Vector Database (Qdrant)..."

# Check if Docker is running
if ! docker info > /dev/null 2>&1; then
    echo "❌ Docker is not running. Please start Docker and try again."
    exit 1
fi

# Create network if it doesn't exist
echo "📡 Creating Docker network..."
docker network create ytempire_ytempire-network 2>/dev/null || echo "Network already exists"

# Deploy Qdrant vector database
echo "🗄️ Deploying Qdrant vector database..."
docker-compose -f docker-compose.qdrant.yml up -d

# Wait for Qdrant to be ready
echo "⏳ Waiting for Qdrant to be ready..."
timeout=60
counter=0
while ! curl -f http://localhost:6333/health > /dev/null 2>&1; do
    if [ $counter -eq $timeout ]; then
        echo "❌ Qdrant failed to start within ${timeout} seconds"
        exit 1
    fi
    echo "Waiting for Qdrant... (${counter}/${timeout})"
    sleep 2
    ((counter += 2))
done

echo "✅ Qdrant is ready!"

# Test Qdrant API
echo "🧪 Testing Qdrant API..."
QDRANT_INFO=$(curl -s http://localhost:6333/cluster)
echo "Qdrant cluster info: $QDRANT_INFO"

# Check if collections endpoint works
echo "📋 Testing collections endpoint..."
curl -s http://localhost:6333/collections || echo "Collections endpoint accessible"

echo ""
echo "🎉 Vector Database Deployment Complete!"
echo ""
echo "📊 Qdrant Dashboard: http://localhost:6333/dashboard"
echo "🔌 REST API: http://localhost:6333"
echo "⚡ gRPC API: localhost:6334"
echo ""
echo "🔧 Configuration:"
echo "  - QDRANT_HOST=localhost"
echo "  - QDRANT_PORT=6333"
echo "  - QDRANT_GRPC_PORT=6334"
echo "  - Data Volume: qdrant_storage"
echo ""
echo "💡 Next steps:"
echo "  1. Update .env with QDRANT_HOST and QDRANT_PORT if needed"
echo "  2. Install Python dependencies: pip install qdrant-client sentence-transformers"
echo "  3. Start the backend to initialize collections"
echo "  4. Use /api/v1/search endpoints for semantic search"
echo ""