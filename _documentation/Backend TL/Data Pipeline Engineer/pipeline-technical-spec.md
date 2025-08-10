# Data Pipeline Technical Architecture & System Design

**Document Version**: 3.0 (Consolidated)  
**Date**: January 2025  
**Scope**: Complete Technical Specifications  
**Infrastructure**: Local Server (Ryzen 9 9950X3D, RTX 5090)

---

## System Architecture Overview

### High-Level Pipeline Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                        Input Layer                               │
│            (API Requests, N8N Triggers, Schedules)              │
└──────────────────────┬──────────────────────────────────────────┘
                       │
┌──────────────────────▼──────────────────────────────────────────┐
│                   Queue Management Layer                         │
│           (PostgreSQL + Redis Priority Queue)                    │
│     Capacity: 100 queued | 7 concurrent (3 GPU + 4 CPU)        │
└──────────────────────┬──────────────────────────────────────────┘
                       │
┌──────────────────────▼──────────────────────────────────────────┐
│                Processing Pipeline Layer                         │
│         (GPU/CPU Scheduling, Video Generation, QA)              │
│            Target: <10 min end-to-end, <$3/video               │
└──────────────────────┬──────────────────────────────────────────┘
                       │
┌──────────────────────▼──────────────────────────────────────────┐
│                Output & Analytics Layer                          │
│       (Upload, Metrics, Cost Tracking, Reporting)               │
│              Real-time dashboards & monitoring                   │
└─────────────────────────────────────────────────────────────────┘
```

## Technology Stack

### Core Technologies (MVP - Confirmed)

```yaml
# Production Stack for 12-week MVP
languages:
  primary: Python 3.11+
  secondary: SQL, Bash
  frontend_integration: JavaScript (WebSocket)

databases:
  postgresql: 
    version: "14.x"
    purpose: "Primary datastore, queue persistence"
  redis:
    version: "7.x"
    purpose: "Cache, real-time queue, pub/sub"
  timescaledb:
    version: "2.x"
    purpose: "Time-series analytics (Week 8+)"

frameworks:
  fastapi: "0.100+ - REST API & WebSocket"
  celery: "5.3+ - Task queue"
  sqlalchemy: "2.0+ - ORM"
  asyncpg: "0.28+ - Async PostgreSQL"
  aioredis: "2.0+ - Async Redis"

orchestration:
  n8n: "Primary workflow automation"
  docker: "Container runtime"
  docker_compose: "Local orchestration"

monitoring:
  prometheus: "Metrics collection"
  grafana: "Visualization"
  custom_dashboards: "Business metrics"

future_considerations:
  # Post-MVP technologies for reference
  streaming: ["Apache Kafka", "Redis Streams"]
  processing: ["Apache Spark", "Apache Beam"]
  cloud: ["Kubernetes", "AWS/GCP services"]
```

## Hardware Specifications

```yaml
local_server:
  cpu:
    model: "AMD Ryzen 9 9950X3D"
    cores: 16
    threads: 32
    allocation:
      system: 4 cores
      postgresql: 4 cores
      pipeline: 8 cores
      reserve: 4 cores
      
  memory:
    total: "128GB DDR5"
    allocation:
      system: 16GB
      postgresql: 16GB
      redis: 8GB
      pipeline_processing: 64GB
      video_buffering: 24GB
      
  gpu:
    model: "NVIDIA RTX 5090"
    vram: "32GB GDDR7"
    allocation:
      model_inference: 8GB
      video_rendering: 20GB
      system_buffer: 4GB
    compute_capability: 9.0
    concurrent_videos:
      simple: 4 max
      complex: 3 max
      
  storage:
    nvme_primary: 
      size: 2TB
      purpose: "OS, hot data, active processing"
    ssd_secondary:
      size: 4TB
      purpose: "Warm data, recent videos"
    hdd_archive:
      size: 10TB
      purpose: "Cold storage, backups"
    
capacity_planning:
  mvp_phase:
    daily_videos: 50
    storage_per_video: 905MB
    daily_storage: 45GB
    retention: 30 days
    
  scale_phase:
    daily_videos: 500
    storage_per_video: 905MB
    daily_storage: 450GB
    retention: 7 days active, 30 days archive
```

## Pipeline Components Design

### 1. Queue Management System

```python
class QueueArchitecture:
    """
    Priority-based queue system with fair scheduling
    """
    
    queue_types = {
        "priority": {
            "storage": "PostgreSQL",
            "index": "Redis sorted set",
            "capacity": 100,
            "ordering": "priority DESC, created_at ASC"
        },
        "processing": {
            "gpu_queue": 3,  # Max concurrent
            "cpu_queue": 4,  # Max concurrent
            "total": 7       # System max
        },
        "retry": {
            "max_attempts": 3,
            "backoff": "exponential",
            "max_delay": 300  # seconds
        }
    }
    
    performance_targets = {
        "enqueue_latency": "<50ms",
        "dequeue_latency": "<100ms",
        "queue_depth_warning": 80,
        "queue_depth_critical": 95
    }
```

### 2. Video Processing Pipeline

```python
class VideoProcessingArchitecture:
    """
    End-to-end video processing with cost control
    """
    
    stages = [
        {
            "name": "script_generation",
            "service": "OpenAI GPT-3.5/4",
            "timeout": 60,
            "cost": 0.40,
            "retry": 2
        },
        {
            "name": "audio_synthesis",
            "service": "Google TTS/ElevenLabs",
            "timeout": 120,
            "cost": 0.20,
            "retry": 2
        },
        {
            "name": "media_collection",
            "service": "Pexels/Pixabay API",
            "timeout": 180,
            "cost": 0.10,
            "retry": 3
        },
        {
            "name": "video_rendering",
            "service": "FFmpeg + GPU/CPU",
            "timeout": 300,
            "cost": 0.30,
            "retry": 1
        },
        {
            "name": "quality_validation",
            "service": "Internal QA",
            "timeout": 30,
            "cost": 0.05,
            "retry": 1
        },
        {
            "name": "upload_preparation",
            "service": "YouTube API",
            "timeout": 120,
            "cost": 0.10,
            "retry": 2
        }
    ]
    
    routing_logic = {
        "simple_video": ["cpu_rendering", "basic_effects"],
        "complex_video": ["gpu_rendering", "advanced_effects"],
        "premium_video": ["gpu_rendering", "ai_enhancement"]
    }
```

### 3. Analytics Pipeline

```python
class AnalyticsArchitecture:
    """
    Real-time and batch analytics processing
    """
    
    data_sources = {
        "youtube_api": {
            "frequency": "hourly",
            "metrics": ["views", "engagement", "revenue"],
            "quota_cost": 100
        },
        "internal_events": {
            "frequency": "real-time",
            "volume": "10K events/hour (MVP)",
            "scaling": "1M+ events/hour (future)"
        },
        "cost_tracking": {
            "frequency": "real-time",
            "precision": "microsecond",
            "aggregation": "per-video, per-service"
        }
    }
    
    processing_requirements = {
        "real_time": {
            "latency": "<1 second",
            "throughput": "10K events/second"
        },
        "batch": {
            "frequency": "hourly/daily",
            "processing_time": "<10 minutes"
        },
        "storage": {
            "compression": "90% reduction",
            "retention": "90 days detailed, 1 year aggregated"
        }
    }
```

### 4. Cost Management System

```python
class CostArchitecture:
    """
    Real-time cost tracking and control
    """
    
    cost_model = {
        "services": {
            "openai_api": {"base": 0.40, "variable": True},
            "tts_service": {"base": 0.20, "variable": True},
            "gpu_compute": {"base": 0.30, "variable": False},
            "storage": {"base": 0.10, "variable": True},
            "api_calls": {"base": 0.15, "variable": True},
            "overhead": {"base": 0.20, "variable": False}
        },
        "targets": {
            "hard_limit": 3.00,      # Never exceed
            "operational": 2.50,     # Target for operations
            "optimal": 2.00,         # Optimization goal
            "stretch": 1.00          # Long-term aspiration
        },
        "alerts": {
            "warning": 2.50,         # 83% of hard limit
            "critical": 2.80,        # 93% of hard limit
            "halt": 3.00            # Stop processing
        }
    }
    
    optimization_strategies = {
        "batching": "30% cost reduction target",
        "caching": "20% API call reduction",
        "scheduling": "Off-peak processing",
        "quality_tiers": "Simple vs Complex routing"
    }
```

## Database Schema

```sql
-- Core queue table
CREATE TABLE video_queue (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    user_id UUID NOT NULL,
    channel_id UUID NOT NULL,
    
    -- Queue management
    priority INTEGER DEFAULT 5 CHECK (priority BETWEEN 1 AND 10),
    status VARCHAR(20) DEFAULT 'queued',
    complexity VARCHAR(20) DEFAULT 'simple',
    
    -- Request data
    request_data JSONB NOT NULL,
    metadata JSONB DEFAULT '{}',
    
    -- Cost tracking
    estimated_cost DECIMAL(10,2) DEFAULT 2.50,
    actual_cost DECIMAL(10,2),
    cost_breakdown JSONB,
    
    -- Timing
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    processing_started_at TIMESTAMP WITH TIME ZONE,
    completed_at TIMESTAMP WITH TIME ZONE,
    
    -- Performance
    processing_time_seconds INTEGER,
    retry_count INTEGER DEFAULT 0,
    error_log JSONB,
    
    CONSTRAINT valid_status CHECK (
        status IN ('queued', 'processing', 'completed', 'failed', 'cancelled')
    ),
    CONSTRAINT valid_complexity CHECK (
        complexity IN ('simple', 'complex', 'premium')
    )
);

-- Indexes for performance
CREATE INDEX idx_queue_priority ON video_queue(status, priority DESC, created_at ASC);
CREATE INDEX idx_queue_channel ON video_queue(channel_id, created_at DESC);
CREATE INDEX idx_queue_status ON video_queue(status) WHERE status IN ('queued', 'processing');
CREATE INDEX idx_queue_cost ON video_queue(actual_cost) WHERE actual_cost IS NOT NULL;

-- Cost tracking table
CREATE TABLE pipeline_costs (
    id BIGSERIAL PRIMARY KEY,
    video_id UUID REFERENCES video_queue(id),
    
    -- Service details
    service VARCHAR(50) NOT NULL,
    operation VARCHAR(100) NOT NULL,
    
    -- Cost data
    amount DECIMAL(10,4) NOT NULL,
    currency VARCHAR(3) DEFAULT 'USD',
    
    -- Metadata
    timestamp TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    metadata JSONB DEFAULT '{}',
    
    CONSTRAINT positive_amount CHECK (amount >= 0)
);

CREATE INDEX idx_costs_video ON pipeline_costs(video_id);
CREATE INDEX idx_costs_service ON pipeline_costs(service, timestamp DESC);
```

## API Specifications

### Pipeline Control APIs

```python
# FastAPI endpoints for pipeline control
from fastapi import FastAPI, WebSocket
from typing import Optional, Dict, Any

app = FastAPI(title="YTEMPIRE Pipeline API")

@app.post("/pipeline/queue")
async def enqueue_video(request: VideoRequest) -> Dict:
    """
    Add video to processing queue
    Returns: {"video_id": "uuid", "position": 5, "eta": 600}
    """
    
@app.get("/pipeline/status/{video_id}")
async def get_status(video_id: str) -> Dict:
    """
    Get current processing status
    Returns: {"status": "processing", "stage": "rendering", "progress": 65}
    """
    
@app.websocket("/pipeline/stream/{video_id}")
async def stream_progress(websocket: WebSocket, video_id: str):
    """
    WebSocket for real-time progress updates
    Sends: {"stage": "rendering", "progress": 65, "eta": 120}
    """
    
@app.get("/pipeline/metrics")
async def get_metrics() -> Dict:
    """
    Get pipeline performance metrics
    Returns: current queue depth, processing rate, error rate
    """
```

## Integration Points

### N8N Workflow Integration

```yaml
n8n_workflows:
  video_processing:
    trigger: "API webhook or schedule"
    nodes:
      - webhook_receiver
      - queue_manager
      - script_generator
      - audio_processor
      - video_renderer
      - youtube_uploader
    error_handling: "Retry with exponential backoff"
    
  cost_tracking:
    trigger: "Every pipeline stage"
    nodes:
      - cost_collector
      - threshold_checker
      - alert_sender
    realtime: true
    
  analytics_collection:
    trigger: "Hourly schedule"
    nodes:
      - youtube_api_fetcher
      - metrics_aggregator
      - dashboard_updater
    batch_size: 50
```

## Security Considerations

```yaml
security_measures:
  api_security:
    - JWT authentication
    - Rate limiting (100 req/min)
    - API key rotation
    
  data_security:
    - Encryption at rest (AES-256)
    - Encryption in transit (TLS 1.3)
    - PII data masking
    
  access_control:
    - Role-based access (RBAC)
    - Audit logging
    - Principle of least privilege
    
  cost_protection:
    - Hard stops at thresholds
    - Automated circuit breakers
    - Real-time alerting
```

## Performance Targets

| Component | Metric | Target | Critical |
|-----------|--------|--------|----------|
| Queue | Dequeue Time | <100ms | >1s |
| Pipeline | End-to-End | <10 min | >15 min |
| API | Response Time | <500ms p95 | >1s |
| Database | Query Time | <150ms p95 | >500ms |
| GPU | Utilization | 70-85% | <50% or >95% |
| Cost | Per Video | <$2.50 | >$3.00 |
| Uptime | System | >99% | <95% |

## Scalability Considerations

### Current (MVP)
- 50 videos/day
- 10K events/hour
- 100 concurrent users
- 45GB daily storage

### Future (Post-MVP)
- 500 videos/day
- 1M+ events/hour
- 1000 concurrent users
- 450GB daily storage

This architecture is designed to scale 10x without major refactoring, with clear upgrade paths identified for each component.