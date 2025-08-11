# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

YTEmpire MVP is an AI-powered YouTube content automation platform built with FastAPI (backend) and React (frontend). The system automates video creation from script generation to publishing, targeting $10,000+ monthly revenue with 95% automation.

## Architecture

### Backend (FastAPI + Python 3.11)
- **Main App**: `backend/app/main.py` - FastAPI application with WebSocket support
- **API Structure**: RESTful API at `/api/v1/` with endpoints for auth, channels, videos, analytics
- **Database**: PostgreSQL 15 with SQLAlchemy async + Alembic migrations
- **Queue System**: Redis + Celery for async video processing tasks
- **AI Services**: OpenAI, Anthropic, ElevenLabs integration in `backend/app/services/`

### Frontend (React 18 + TypeScript + Vite)
- **Framework**: React with TypeScript, Vite build system
- **State Management**: Zustand stores in `frontend/src/stores/`
- **Styling**: Tailwind CSS with Material-UI components
- **Real-time**: WebSocket integration for live updates

### ML Pipeline
- **Configuration**: `ml-pipeline/config.yaml` defines model settings, cost limits, performance targets
- **Services**: Script generation, voice synthesis, thumbnail generation, video assembly
- **Cost Target**: <$3 per video with quality thresholds

## Development Commands

### Full Stack Setup (Recommended)
```bash
# Complete stack with Docker Compose
docker-compose up -d

# Access services:
# - Backend API: http://localhost:8000 (docs at /docs)  
# - Frontend: http://localhost:3000
# - Flower (Celery monitoring): http://localhost:5555
# - N8N workflows: http://localhost:5678
# - Grafana dashboards: http://localhost:3001
# - Prometheus metrics: http://localhost:9090
```

### Backend Development
```bash
cd backend
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate
pip install -r requirements.txt
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

### Frontend Development  
```bash
cd frontend
npm install
npm run dev  # Starts on port 3000
npm run build  # Production build
npm run preview  # Preview production build
```

### Testing
```bash
# Backend tests (pytest configured in pytest.ini)
cd backend
pytest tests/ -v --cov=app
pytest tests/ -m unit  # Unit tests only
pytest tests/ -m integration  # Integration tests only

# Frontend tests (Jest configured in jest.config.js) 
cd frontend
npm test
npm run test:coverage  # With coverage
npm run test -- --watch  # Watch mode for development
```

### Database Operations
```bash
cd backend
# Create migration
alembic revision --autogenerate -m "description"
# Apply migrations  
alembic upgrade head
# Downgrade migration
alembic downgrade -1
```

### Code Quality & Linting
```bash
# Backend linting/formatting
cd backend
black app/
flake8 app/  
mypy app/
pre-commit run --all-files  # Run all pre-commit hooks

# Frontend linting
cd frontend
npm run lint
npm run lint:fix  # Auto-fix issues
npm run typecheck  # TypeScript type checking
```

### N8N Workflow Management
```bash
# N8N development (access at localhost:5678)
# Credentials: admin / ytempire_n8n_2024
# Import workflows from infrastructure/n8n/workflows/
```

## Team-Specific Documentation

**Comprehensive role-based documentation available in `_documentation/` folder:**

- **AI/ML Team Lead**: ML pipeline implementation, model optimization, cost targets
- **Backend Team Lead**: API development, integrations, system architecture
- **Frontend Team Lead**: React components, dashboard development, UI/UX
- **Data Team Lead**: Analytics pipelines, metrics collection, data engineering
- **Platform Ops Team Lead**: DevOps, security, monitoring, deployment

**Weekly Sprint Plans**: Available for weeks 0-3 with detailed task breakdowns and team coordination

## Key Service Patterns

### Video Generation Pipeline
The core business logic follows this pattern:
1. **Trend Analysis** → Script Generation (GPT-4/Claude)
2. **Script Processing** → Voice Synthesis (ElevenLabs)  
3. **Visual Generation** → Thumbnail Creation (DALL-E 3)
4. **Video Assembly** → Final video compilation
5. **Quality Checks** → Publishing to YouTube

### Multi-Agent Orchestration System
- **TrendProphet**: Prophet, LSTM models for trend prediction (85% accuracy target)
- **ContentStrategist**: GPT-4, Claude-2 for script generation (<30s latency, <$0.10/call)
- **QualityGuardian**: BERT-QA, custom scoring models (85% min score)
- **RevenueOptimizer**: Multi-Armed-Bandit, Bayesian optimization (24h cycles)
- **CrisisManager**: Anomaly detection, policy checking (<5min response)
- **NicheExplorer**: Clustering, topic modeling for market discovery

### YouTube Multi-Account Management
- **Account Rotation**: Intelligent health scoring and load balancing across 15 accounts
- **Quota Management**: 10,000 units per account with progressive fallback
- **Compliance Monitoring**: Automated policy checking to prevent strikes
- **Health Scoring**: Real-time account status monitoring

### Cost Tracking & Optimization
- All AI service calls tracked in `backend/app/services/cost_tracking.py`
- **Daily Budget Limits**: OpenAI $50, ElevenLabs $20, Google $10
- **Progressive Model Fallback**: GPT-4 → GPT-3.5 → Claude for cost optimization
- **Aggressive Caching**: 1-hour TTL for trending topics, 15-min for analytics
- Target: <$3 per video, optimization goal $0.50 per video

### WebSocket Real-time Updates
- Video generation progress updates via WebSocket
- Endpoints: `/ws/{client_id}` and `/ws/video-updates/{channel_id}`
- Manager: `backend/app/services/websocket_manager.py`

## Database Schema Key Models
- **User**: Authentication, multi-tenant support
- **Channel**: YouTube channel management (5+ channels per user)
- **Video**: Video metadata, generation status, analytics
- **Cost**: Granular cost tracking per service/video

## Environment Configuration
- Backend settings in `backend/app/core/config.py` with Pydantic validation
- Frontend environment variables via Vite (VITE_API_URL)
- ML pipeline config in `ml-pipeline/config.yaml`

## Security & Performance
- **Authentication**: JWT with RS256, OAuth 2.0 for YouTube (15-account management)
- **Encryption**: AES-256 at rest, TLS 1.3 in transit
- **Access Control**: RBAC with scoped API keys, 2FA optional
- **Rate Limiting**: 1000 requests/hour per user, per-IP protection
- **Performance Monitoring**: Prometheus + Grafana with custom dashboards
- **Caching Strategy**: Redis with service-specific TTL policies
- **Backup & Recovery**: Hourly incremental, daily full (RTO: 4h, RPO: 1h)

## AI Service Integration
Primary services configured with fallbacks:
- **Script Generation**: GPT-4-turbo → Claude-3-opus  
- **Voice Synthesis**: ElevenLabs → Google TTS
- **Thumbnails**: DALL-E 3
- **Quality Scoring**: Custom models for content evaluation

## Monitoring & Alerting
- **Health Checks**: `GET /health` endpoint with service status
- **Metrics**: Prometheus scraping at `/metrics` (30-second intervals)
- **Critical Alerts** (immediate): System down, API error rate >5%, security incidents
- **Warning Alerts** (email): High latency >1s, queue depth >100, cost >$2/video
- **Performance Targets**: <500ms API p95, <10min video generation, >99% uptime
- **Business Metrics**: Sprint velocity, cost optimization (30% reduction goal)

## Hardware Specifications (MVP Phase)
- **CPU**: AMD Ryzen 9 9950X3D (16 cores, 32 threads)
- **RAM**: 128GB DDR5 (PostgreSQL: 16GB, Redis: 8GB, Video processing: 48GB)
- **GPU**: NVIDIA RTX 5090 (32GB VRAM for AI model inference)
- **Storage**: 2TB NVMe system + 4TB NVMe data + 8TB HDD backup

## Common Debugging Commands

### Check Service Health
```bash
# Backend health check
curl http://localhost:8000/health

# Check running containers
docker ps

# View container logs
docker logs ytempire_backend -f
docker logs ytempire_celery_worker -f

# Check Celery task status
celery -A app.core.celery_app inspect active
celery -A app.core.celery_app inspect stats
```

### Running Individual Tests
```bash
# Backend: Run a specific test file
cd backend
pytest tests/test_video_service.py -v

# Backend: Run a specific test function
pytest tests/test_video_service.py::test_generate_video -v

# Frontend: Run tests matching a pattern
cd frontend
npm test -- --testNamePattern="VideoCard"
```

### Environment Variables
```bash
# Backend requires these environment variables (see .env.example):
# DATABASE_URL, REDIS_URL, JWT_SECRET_KEY, OPENAI_API_KEY, ANTHROPIC_API_KEY, 
# ELEVENLABS_API_KEY, YOUTUBE_API_KEY, GOOGLE_CLOUD_PROJECT_ID

# Frontend requires:
# VITE_API_URL (defaults to http://localhost:8000)
```
- always place test or fix scripts that you generate in the misc/ folder
- Always refer to the provided documentation in the _documentation/ folder before executing a development task