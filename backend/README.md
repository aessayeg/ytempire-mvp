# YTEmpire Backend API

## Overview

YTEmpire Backend is a high-performance FastAPI application that powers the AI-driven YouTube content automation platform. Built for scale, it handles video generation, multi-account management, and real-time analytics for 100+ videos per day.

## Architecture

```
backend/
├── app/
│   ├── api/           # API endpoints (42 files, 397 routes)
│   ├── core/          # Core utilities (auth, config, celery)
│   ├── models/        # SQLAlchemy models (39 models)
│   ├── services/      # Business logic (60 services)
│   ├── tasks/         # Celery tasks (59 async tasks)
│   └── main.py        # FastAPI application
├── alembic/           # Database migrations
├── tests/             # Test suites
└── requirements.txt   # Python dependencies
```

## Quick Start

### Prerequisites
- Python 3.11+
- PostgreSQL 15+
- Redis 6+
- Docker & Docker Compose

### Installation

1. **Clone and setup environment:**
```bash
cd backend
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

2. **Configure environment variables:**
```bash
cp .env.example .env
# Edit .env with your configuration
```

3. **Run database migrations:**
```bash
alembic upgrade head
```

4. **Start the server:**
```bash
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

### Docker Setup

```bash
docker-compose up -d backend
```

Access API documentation at: http://localhost:8000/docs

## API Documentation

### Authentication
All endpoints require JWT authentication except `/auth/login` and `/auth/register`.

```bash
# Login
curl -X POST http://localhost:8000/api/v1/auth/login \
  -H "Content-Type: application/json" \
  -d '{"email": "user@example.com", "password": "password"}'

# Use token in headers
curl -H "Authorization: Bearer <token>" http://localhost:8000/api/v1/channels
```

### Core Endpoints

#### Video Generation
- `POST /api/v1/videos/generate` - Generate new video
- `GET /api/v1/videos/{id}/status` - Check generation status
- `POST /api/v1/videos/bulk` - Bulk video generation (10+ videos)

#### Channel Management
- `GET /api/v1/channels` - List user channels
- `POST /api/v1/channels` - Create new channel
- `PUT /api/v1/channels/{id}` - Update channel
- `DELETE /api/v1/channels/{id}` - Delete channel

#### Analytics
- `GET /api/v1/analytics/dashboard` - Dashboard metrics
- `GET /api/v1/analytics/channels/{id}` - Channel performance
- `GET /api/v1/analytics/videos/{id}` - Video analytics
- `GET /api/v1/analytics/costs` - Cost breakdown

#### YouTube Integration
- `GET /api/v1/youtube/accounts` - List connected accounts
- `POST /api/v1/youtube/oauth/init` - Start OAuth flow
- `GET /api/v1/youtube/quota` - Check quota status
- `POST /api/v1/youtube/upload` - Upload video

### WebSocket Support
Real-time updates via WebSocket at `/ws/{client_id}`:

```javascript
const ws = new WebSocket('ws://localhost:8000/ws/client123');
ws.onmessage = (event) => {
  const data = JSON.parse(event.data);
  console.log('Video status:', data.status);
};
```

## Services

### Core Services (60 total)
- **Video Generation Pipeline** - End-to-end video creation
- **Multi-Account Manager** - 15 YouTube account rotation
- **Cost Tracker** - Real-time cost monitoring (<$3/video)
- **Analytics Service** - Performance metrics & reporting
- **AI Integration** - GPT-4, Claude, ElevenLabs, DALL-E
- **WebSocket Manager** - Real-time client updates

## Database Schema

### Key Models
- **User** - Authentication & multi-tenancy
- **Channel** - YouTube channel management
- **Video** - Video metadata & status
- **Cost** - Granular cost tracking
- **Analytics** - Performance metrics

## Celery Tasks

### Task Queues
- `video_generation` - High priority video tasks
- `youtube_upload` - Upload queue with retry
- `analytics` - Background analytics processing
- `maintenance` - Cleanup and optimization

### Running Workers
```bash
celery -A app.core.celery_app worker --loglevel=info --concurrency=4
celery -A app.core.celery_app flower  # Monitoring UI at :5555
```

## Testing

### Run Tests
```bash
# All tests
pytest tests/ -v --cov=app

# Unit tests only
pytest tests/ -m unit

# Integration tests
pytest tests/ -m integration

# Specific test file
pytest tests/test_video_service.py -v
```

### Test Coverage
Current coverage: ~85% (target: 90%)

## Performance

### Benchmarks
- API Response: <200ms p95
- Video Generation: <10 minutes
- Concurrent Videos: 10+ 
- Daily Capacity: 100+ videos
- Cost per Video: $2.04 average

### Optimization
- Redis caching (15min TTL)
- Connection pooling
- Async I/O throughout
- Batch processing support

## Monitoring

### Health Check
```bash
curl http://localhost:8000/health
```

### Metrics
Prometheus metrics available at `/metrics`:
- Request latency
- Error rates
- Queue depths
- Resource usage

## Security

- JWT with RS256 signing
- OAuth 2.0 for YouTube
- Rate limiting (1000 req/hour)
- Input validation
- SQL injection protection
- XSS prevention

## Deployment

### Production
```bash
docker-compose -f docker-compose.production.yml up -d
```

### Environment Variables
Required variables in production:
- `DATABASE_URL` - PostgreSQL connection
- `REDIS_URL` - Redis connection
- `JWT_SECRET_KEY` - JWT signing key
- `OPENAI_API_KEY` - OpenAI API
- `YOUTUBE_API_KEY` - YouTube Data API
- `ELEVENLABS_API_KEY` - Voice synthesis

## Contributing

1. Create feature branch
2. Write tests (maintain >85% coverage)
3. Update documentation
4. Submit PR with description

## Support

- API Docs: http://localhost:8000/docs
- Logs: `docker logs ytempire_backend`
- Issues: GitHub Issues

## License

Proprietary - YTEmpire © 2024