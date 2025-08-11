#!/bin/bash

# YTEmpire AI/ML Content Quality Scoring Setup Script
# Sets up comprehensive quality analysis system for video content

set -euo pipefail

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Configuration
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
LOG_FILE="/var/log/ytempire-quality-setup.log"

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
    log "Checking prerequisites for quality scoring setup..."
    
    # Check Python
    if ! command -v python3 &> /dev/null; then
        error "Python 3 is not installed. Please install Python 3.11 or higher."
    fi
    
    # Check Python version
    PYTHON_VERSION=$(python3 -c "import sys; print('.'.join(map(str, sys.version_info[:2])))")
    if [[ $(echo "$PYTHON_VERSION < 3.9" | bc -l) -eq 1 ]]; then
        error "Python 3.9 or higher required. Found: $PYTHON_VERSION"
    fi
    
    # Check Docker
    if ! command -v docker &> /dev/null; then
        error "Docker is not installed. Please install Docker first."
    fi
    
    # Check Docker Compose
    if ! command -v docker-compose &> /dev/null; then
        error "Docker Compose is not installed. Please install Docker Compose first."
    fi
    
    # Check FFmpeg
    if ! command -v ffmpeg &> /dev/null; then
        error "FFmpeg is not installed. Please install FFmpeg first."
    fi
    
    # Check available disk space
    AVAILABLE_SPACE=$(df "$PROJECT_ROOT" | tail -1 | awk '{print $4}')
    REQUIRED_SPACE=10485760  # 10GB in KB
    
    if [[ $AVAILABLE_SPACE -lt $REQUIRED_SPACE ]]; then
        warning "Less than 10GB available disk space. ML models may require more space."
    fi
    
    # Check GPU availability
    if command -v nvidia-smi &> /dev/null; then
        success "NVIDIA GPU detected - GPU acceleration available"
        GPU_AVAILABLE=true
    else
        warning "No NVIDIA GPU detected - using CPU-only mode"
        GPU_AVAILABLE=false
    fi
    
    success "Prerequisites check completed"
}

# Setup Python virtual environment
setup_python_environment() {
    log "Setting up Python virtual environment for quality scoring..."
    
    cd "$PROJECT_ROOT/ml-pipeline/quality_scoring"
    
    # Create virtual environment if it doesn't exist
    if [[ ! -d "venv" ]]; then
        python3 -m venv venv
        success "Virtual environment created"
    else
        log "Virtual environment already exists"
    fi
    
    # Activate virtual environment
    source venv/bin/activate
    
    # Upgrade pip
    pip install --upgrade pip
    
    # Install requirements
    log "Installing Python dependencies..."
    pip install -r requirements.txt
    
    # Install additional dependencies for GPU if available
    if [[ "$GPU_AVAILABLE" == "true" ]]; then
        log "Installing GPU-optimized packages..."
        pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
    fi
    
    success "Python environment setup completed"
}

# Download and setup ML models
setup_ml_models() {
    log "Setting up ML models for quality scoring..."
    
    cd "$PROJECT_ROOT/ml-pipeline/quality_scoring"
    source venv/bin/activate
    
    # Create models directory
    mkdir -p models/
    
    # Download and cache models
    log "Downloading Whisper models..."
    python3 -c "
import whisper
whisper.load_model('base')
print('Whisper model downloaded')
"
    
    log "Downloading CLIP models..."
    python3 -c "
from transformers import CLIPProcessor, CLIPModel
processor = CLIPProcessor.from_pretrained('openai/clip-vit-base-patch32')
model = CLIPModel.from_pretrained('openai/clip-vit-base-patch32')
print('CLIP model downloaded')
"
    
    log "Downloading BERT models..."
    python3 -c "
from transformers import BertTokenizer, BertModel
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
model = BertModel.from_pretrained('bert-base-uncased')
print('BERT model downloaded')
"
    
    log "Downloading sentiment analysis models..."
    python3 -c "
from transformers import pipeline
analyzer = pipeline('sentiment-analysis', model='cardiffnlp/twitter-roberta-base-sentiment-latest')
print('Sentiment analysis model downloaded')
"
    
    log "Downloading topic classification models..."
    python3 -c "
from transformers import pipeline
classifier = pipeline('zero-shot-classification', model='facebook/bart-large-mnli')
print('Topic classification model downloaded')
"
    
    # Download NLTK data
    log "Downloading NLTK data..."
    python3 -c "
import nltk
nltk.download('punkt', quiet=True)
nltk.download('averaged_perceptron_tagger', quiet=True)
print('NLTK data downloaded')
"
    
    success "ML models setup completed"
}

# Setup database for quality scores
setup_database() {
    log "Setting up database for quality scores..."
    
    cd "$PROJECT_ROOT"
    
    # Create database initialization script
    cat > ml-pipeline/quality_scoring/init-db.sql << 'EOF'
-- Initialize Quality Scoring Database
CREATE EXTENSION IF NOT EXISTS "uuid-ossp";

-- Quality task results table
CREATE TABLE IF NOT EXISTS quality_task_results (
    id SERIAL PRIMARY KEY,
    task_id VARCHAR(255) UNIQUE NOT NULL,
    video_path TEXT NOT NULL,
    status VARCHAR(50) NOT NULL,
    overall_score FLOAT,
    processing_time FLOAT,
    error_message TEXT,
    metrics_json TEXT,
    report_json TEXT,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    completed_at TIMESTAMP
);

-- Quality scores cache table
CREATE TABLE IF NOT EXISTS quality_scores_cache (
    id SERIAL PRIMARY KEY,
    video_hash VARCHAR(255) UNIQUE NOT NULL,
    video_path TEXT NOT NULL,
    analysis_timestamp TIMESTAMP NOT NULL,
    metrics_json TEXT NOT NULL,
    overall_score FLOAT NOT NULL,
    processing_time FLOAT NOT NULL,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Indexes for performance
CREATE INDEX IF NOT EXISTS idx_task_results_task_id ON quality_task_results(task_id);
CREATE INDEX IF NOT EXISTS idx_task_results_status ON quality_task_results(status);
CREATE INDEX IF NOT EXISTS idx_task_results_created_at ON quality_task_results(created_at);
CREATE INDEX IF NOT EXISTS idx_quality_cache_hash ON quality_scores_cache(video_hash);
CREATE INDEX IF NOT EXISTS idx_quality_cache_score ON quality_scores_cache(overall_score);

-- Quality statistics view
CREATE OR REPLACE VIEW quality_statistics AS
SELECT 
    COUNT(*) as total_analyses,
    AVG(overall_score) as average_score,
    MIN(overall_score) as min_score,
    MAX(overall_score) as max_score,
    COUNT(CASE WHEN created_at > NOW() - INTERVAL '24 hours' THEN 1 END) as recent_analyses,
    COUNT(CASE WHEN status = 'completed' THEN 1 END) as completed_analyses,
    COUNT(CASE WHEN status = 'failed' THEN 1 END) as failed_analyses
FROM quality_task_results;

-- Grant permissions
GRANT ALL PRIVILEGES ON ALL TABLES IN SCHEMA public TO quality_user;
GRANT ALL PRIVILEGES ON ALL SEQUENCES IN SCHEMA public TO quality_user;
EOF

    success "Database schema created"
}

# Setup configuration files
setup_configuration() {
    log "Setting up configuration files..."
    
    cd "$PROJECT_ROOT"
    
    # Update configuration for current environment
    if [[ "$GPU_AVAILABLE" == "true" ]]; then
        # Enable GPU acceleration
        sed -i 's/device: "auto"/device: "cuda"/' ml-pipeline/quality_scoring/config.yaml
        success "GPU acceleration enabled in configuration"
    else
        # Use CPU-only mode
        sed -i 's/device: "auto"/device: "cpu"/' ml-pipeline/quality_scoring/config.yaml
        log "CPU-only mode configured"
    fi
    
    # Create environment file
    cat > ml-pipeline/quality_scoring/.env << EOF
# Quality Scoring Environment Variables

# API Configuration
QUALITY_API_HOST=0.0.0.0
QUALITY_API_PORT=8001
QUALITY_API_WORKERS=1

# Database Configuration
DATABASE_URL=postgresql://quality_user:quality_secure_password_2024@localhost:5433/quality_scores

# Redis Configuration
CELERY_BROKER_URL=redis://:quality_redis_password_2024@localhost:6380/0
CELERY_RESULT_BACKEND=redis://:quality_redis_password_2024@localhost:6380/0

# Model Configuration
WHISPER_MODEL_SIZE=base
CUDA_VISIBLE_DEVICES=0

# Processing Configuration
MAX_CONCURRENT_ANALYSES=4
TARGET_PROCESSING_TIME=120
ENABLE_CACHE=true
CACHE_DURATION_HOURS=24

# Logging
LOG_LEVEL=INFO
LOG_FILE=/var/log/ytempire/quality_scoring.log

# Security
SECRET_KEY=$(openssl rand -hex 32)

# Development
DEBUG_MODE=false
ENABLE_PROFILING=false
EOF

    success "Configuration files created"
}

# Setup monitoring configuration
setup_monitoring() {
    log "Setting up monitoring configuration..."
    
    cd "$PROJECT_ROOT"
    
    # Create monitoring directory
    mkdir -p monitoring/
    
    # Prometheus configuration for quality scoring
    cat > monitoring/quality-prometheus.yml << 'EOF'
global:
  scrape_interval: 15s
  evaluation_interval: 15s

rule_files:
  - "quality_rules.yml"

scrape_configs:
  - job_name: 'quality-scoring'
    static_configs:
      - targets: ['quality-scorer:8001']
    metrics_path: '/metrics'
    scrape_interval: 30s

  - job_name: 'quality-worker'
    static_configs:
      - targets: ['quality-worker:9540']
    metrics_path: '/metrics'
    scrape_interval: 30s

alerting:
  alertmanagers:
    - static_configs:
        - targets:
          - alertmanager:9093
EOF

    # Grafana datasources configuration
    cat > monitoring/grafana-datasources.yml << 'EOF'
apiVersion: 1

datasources:
  - name: Prometheus
    type: prometheus
    access: proxy
    url: http://quality-monitor:9090
    isDefault: true
EOF

    # Create Grafana dashboard directory
    mkdir -p monitoring/grafana-quality-dashboards/
    
    # Quality scoring dashboard
    cat > monitoring/grafana-quality-dashboards/quality-dashboard.json << 'EOF'
{
  "dashboard": {
    "id": null,
    "title": "YTEmpire Quality Scoring",
    "tags": ["ytempire", "quality", "ml"],
    "timezone": "browser",
    "panels": [
      {
        "id": 1,
        "title": "Average Quality Score",
        "type": "stat",
        "targets": [
          {
            "expr": "avg(quality_score_average)",
            "format": "time_series"
          }
        ]
      },
      {
        "id": 2,
        "title": "Processing Time Distribution",
        "type": "histogram",
        "targets": [
          {
            "expr": "histogram_quantile(0.95, quality_processing_time_bucket)",
            "format": "time_series"
          }
        ]
      },
      {
        "id": 3,
        "title": "Analysis Success Rate",
        "type": "graph",
        "targets": [
          {
            "expr": "rate(quality_analysis_success_total[5m]) / rate(quality_analysis_total[5m])",
            "format": "time_series"
          }
        ]
      }
    ],
    "time": {
      "from": "now-1h",
      "to": "now"
    },
    "refresh": "30s"
  }
}
EOF

    success "Monitoring configuration setup completed"
}

# Create test data and scripts
create_test_data() {
    log "Creating test data and scripts..."
    
    cd "$PROJECT_ROOT"
    
    # Create test directory
    mkdir -p test_videos/
    mkdir -p ml-pipeline/quality_scoring/test/
    
    # Create test script
    cat > ml-pipeline/quality_scoring/test_quality_scorer.py << 'EOF'
#!/usr/bin/env python3
"""
Test script for quality scoring system
"""

import asyncio
import os
import sys
from pathlib import Path

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent.parent))

from ml_pipeline.quality_scoring.quality_scorer import (
    ContentQualityScorer, QualityScoringConfig
)

async def test_quality_scorer():
    """Test quality scoring functionality"""
    
    # Configuration
    config = QualityScoringConfig(
        min_acceptable_score=0.6,
        max_concurrent_analyses=2
    )
    
    # Initialize scorer
    scorer = ContentQualityScorer(config)
    
    try:
        # Test with sample data
        script_text = "Welcome to our channel! Today we'll explore artificial intelligence and machine learning."
        target_topic = "technology"
        
        print("Testing content analysis...")
        content_metrics = await scorer.content_analyzer.analyze_content_quality(
            script_text, target_topic
        )
        print(f"Content metrics: {content_metrics}")
        
        # Test database operations
        print("Testing database operations...")
        stats = scorer.database.get_statistics()
        print(f"Database stats: {stats}")
        
        print("Quality scorer test completed successfully!")
        
    finally:
        scorer.close()

if __name__ == "__main__":
    asyncio.run(test_quality_scorer())
EOF

    chmod +x ml-pipeline/quality_scoring/test_quality_scorer.py
    
    # Create sample video generation script (placeholder)
    cat > scripts/ml/generate-test-video.py << 'EOF'
#!/usr/bin/env python3
"""
Generate test video for quality scoring
"""

import cv2
import numpy as np
import os
from pathlib import Path

def generate_test_video(output_path: str, duration: int = 30):
    """Generate a simple test video"""
    
    # Video parameters
    width, height = 1280, 720
    fps = 30
    total_frames = duration * fps
    
    # Create video writer
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
    
    for frame_num in range(total_frames):
        # Create frame with changing colors
        hue = int((frame_num / total_frames) * 180)
        frame = np.full((height, width, 3), [hue, 255, 255], dtype=np.uint8)
        frame = cv2.cvtColor(frame, cv2.COLOR_HSV2BGR)
        
        # Add text
        text = f"Frame {frame_num + 1}/{total_frames}"
        cv2.putText(frame, text, (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        
        out.write(frame)
    
    out.release()
    print(f"Test video generated: {output_path}")

if __name__ == "__main__":
    output_dir = Path(__file__).parent.parent.parent / "test_videos"
    output_dir.mkdir(exist_ok=True)
    
    output_path = output_dir / "test_video.mp4"
    generate_test_video(str(output_path))
EOF

    chmod +x scripts/ml/generate-test-video.py
    
    success "Test data and scripts created"
}

# Setup Docker services
setup_docker_services() {
    log "Setting up Docker services for quality scoring..."
    
    cd "$PROJECT_ROOT"
    
    # Build quality scoring service
    log "Building quality scoring Docker image..."
    docker build -t ytempire-quality-scorer ml-pipeline/quality_scoring/
    
    # Start services
    log "Starting quality scoring services..."
    docker-compose -f docker-compose.quality.yml up -d
    
    # Wait for services to be healthy
    log "Waiting for services to become healthy..."
    sleep 30
    
    # Check service health
    if docker ps | grep -q "ytempire_quality_scorer"; then
        success "Quality scorer service is running"
    else
        error "Quality scorer service failed to start"
    fi
    
    if docker ps | grep -q "ytempire_quality_db"; then
        success "Quality database service is running"
    else
        error "Quality database service failed to start"
    fi
    
    success "Docker services setup completed"
}

# Validate installation
validate_installation() {
    log "Validating quality scoring installation..."
    
    # Test API endpoint
    if curl -f http://localhost:8001/health > /dev/null 2>&1; then
        success "Quality scoring API is responding"
    else
        warning "Quality scoring API is not responding on port 8001"
    fi
    
    # Test database connection
    if docker exec ytempire_quality_db pg_isready -U quality_user -d quality_scores > /dev/null 2>&1; then
        success "Database connection is working"
    else
        warning "Database connection test failed"
    fi
    
    # Test Python environment
    cd "$PROJECT_ROOT/ml-pipeline/quality_scoring"
    if source venv/bin/activate && python3 test_quality_scorer.py > /dev/null 2>&1; then
        success "Python environment test passed"
    else
        warning "Python environment test failed"
    fi
    
    # Check model files
    MODELS_DIR="$PROJECT_ROOT/ml-pipeline/quality_scoring/venv/lib/python*/site-packages/transformers/models"
    if [[ -d "$MODELS_DIR" ]] && [[ $(find "$MODELS_DIR" -type f | wc -l) -gt 0 ]]; then
        success "ML models are downloaded"
    else
        warning "ML models may not be properly downloaded"
    fi
    
    log "Installation validation completed"
}

# Print setup instructions
print_instructions() {
    log "Quality scoring setup completed successfully!"
    
    cat << EOF

${GREEN}YTEmpire AI/ML Content Quality Scoring Setup Complete!${NC}

${YELLOW}Services Status:${NC}
- Quality Scorer API: http://localhost:8001
- Quality Database: localhost:5433
- Quality Redis: localhost:6380
- Grafana Dashboard: http://localhost:3001
- Prometheus Metrics: http://localhost:9091

${YELLOW}API Endpoints:${NC}
- Health Check: GET http://localhost:8001/health
- Analyze Video: POST http://localhost:8001/analyze
- Batch Analysis: POST http://localhost:8001/analyze/batch
- Upload & Analyze: POST http://localhost:8001/upload-and-analyze
- Statistics: GET http://localhost:8001/stats
- Configuration: GET http://localhost:8001/config

${YELLOW}Usage Examples:${NC}

1. Test quality analysis:
   cd ${PROJECT_ROOT}/ml-pipeline/quality_scoring
   source venv/bin/activate
   python3 test_quality_scorer.py

2. Generate test video:
   python3 ${PROJECT_ROOT}/scripts/ml/generate-test-video.py

3. Analyze video via API:
   curl -X POST "http://localhost:8001/analyze" \\
     -H "Content-Type: application/json" \\
     -d '{
       "video_path": "/path/to/video.mp4",
       "script_text": "Your video script here",
       "target_topic": "technology"
     }'

4. Check service health:
   curl http://localhost:8001/health

5. View statistics:
   curl http://localhost:8001/stats

${YELLOW}Quality Metrics Included:${NC}
- ✅ Visual Quality Analysis (resolution, sharpness, brightness, contrast)
- ✅ Audio Quality Analysis (clarity, noise level, speech quality)
- ✅ Content Analysis (coherence, relevance, engagement potential)
- ✅ Technical Analysis (encoding quality, file size efficiency)
- ✅ Composite Scoring with confidence intervals
- ✅ Engagement and monetization potential prediction

${YELLOW}ML Models Integrated:${NC}
- OpenAI Whisper for speech recognition and quality
- OpenAI CLIP for visual-semantic analysis
- BERT for content coherence analysis
- RoBERTa for sentiment analysis
- BART for topic classification

${YELLOW}Management Commands:${NC}
- Start services: docker-compose -f docker-compose.quality.yml up -d
- Stop services: docker-compose -f docker-compose.quality.yml down
- View logs: docker-compose -f docker-compose.quality.yml logs -f
- Scale workers: docker-compose -f docker-compose.quality.yml up --scale quality-worker=3 -d

${YELLOW}Monitoring:${NC}
- Grafana Dashboard: http://localhost:3001 (admin/quality_grafana_admin_2024)
- Prometheus Metrics: http://localhost:9091
- Service Logs: /var/log/ytempire/quality_scoring.log

${GREEN}Your AI/ML quality scoring system is now ready for video analysis!${NC}

EOF
}

# Main execution
main() {
    log "Starting YTEmpire AI/ML Quality Scoring Setup"
    
    check_prerequisites
    setup_python_environment
    setup_ml_models
    setup_database
    setup_configuration
    setup_monitoring
    create_test_data
    setup_docker_services
    validate_installation
    
    print_instructions
    
    success "AI/ML Quality scoring setup completed successfully!"
}

# Run main function with all arguments
main "$@"