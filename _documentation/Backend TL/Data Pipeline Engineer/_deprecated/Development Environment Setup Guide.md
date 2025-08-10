# Development Environment Setup Guide

**Document Version**: 1.0  
**Date**: January 2025  
**For**: API Development Engineer  
**Classification**: Environment Configuration

---

## 📦 Table of Contents

1. [Environment Variables Configuration](#environment-variables-configuration)
2. [Docker Setup](#docker-setup)
3. [Database Initialization](#database-initialization)
4. [Local Development Setup](#local-development-setup)
5. [External Services Setup](#external-services-setup)
6. [Testing Environment](#testing-environment)

---

## Environment Variables Configuration

### Complete .env Template

```bash
# .env file for YTEMPIRE Backend

# === APPLICATION SETTINGS ===
APP_NAME=YTEMPIRE
APP_VERSION=1.0.0
DEBUG=true
LOG_LEVEL=INFO
API_BASE_URL=http://localhost:8000

# === SERVER CONFIGURATION ===
HOST=0.0.0.0
PORT=8000
WORKERS=4
RELOAD=true  # For development only

# === DATABASE CONFIGURATION ===
DATABASE_URL=postgresql+asyncpg://ytempire:ytempire_pass@localhost:5432/ytempire_db
DATABASE_POOL_SIZE=20
DATABASE_MAX_OVERFLOW=40
DATABASE_ECHO=false  # Set to true for SQL logging

# === REDIS CONFIGURATION ===
REDIS_URL=redis://localhost:6379/0
REDIS_TTL=3600
REDIS_MAX_CONNECTIONS=50

# === SECURITY SETTINGS ===
SECRET_KEY=your-super-secret-key-change-this-in-production-minimum-32-chars
ALGORITHM=HS256
ACCESS_TOKEN_EXPIRE_MINUTES=30
REFRESH_TOKEN_EXPIRE_DAYS=7
INTERNAL_API_KEY=internal-service-communication-key

# === YOUTUBE API CONFIGURATION ===
YOUTUBE_CLIENT_ID=your-client-id.apps.googleusercontent.com
YOUTUBE_CLIENT_SECRET=your-client-secret
YOUTUBE_REDIRECT_URI=http://localhost:8000/api/v1/auth/youtube/callback
YOUTUBE_ACCOUNTS=15
YOUTUBE_DAILY_QUOTA=10000
YOUTUBE_VIDEOS_PER_ACCOUNT=5
YOUTUBE_API_KEY=your-youtube-api-key  # For non-OAuth operations

# === OPENAI CONFIGURATION ===
OPENAI_API_KEY=sk-proj-your-openai-api-key
OPENAI_MODEL=gpt-3.5-turbo
OPENAI_MAX_TOKENS=2000
OPENAI_TEMPERATURE=0.7
OPENAI_ORGANIZATION=org-your-org-id  # Optional

# === GOOGLE TEXT-TO-SPEECH ===
GOOGLE_TTS_KEY=your-google-cloud-api-key
GOOGLE_TTS_LANGUAGE=en-US
GOOGLE_TTS_VOICE=en-US-Neural2-J
GOOGLE_CLOUD_PROJECT=your-project-id

# === ELEVENLABS CONFIGURATION (Optional Fallback) ===
ELEVENLABS_API_KEY=your-elevenlabs-api-key
ELEVENLABS_VOICE_ID=21m00Tcm4TlvDq8ikWAM

# === STRIPE CONFIGURATION ===
STRIPE_SECRET_KEY=sk_test_your-stripe-secret-key
STRIPE_PUBLISHABLE_KEY=pk_test_your-stripe-publishable-key
STRIPE_WEBHOOK_SECRET=whsec_your-webhook-secret
STRIPE_PRICE_ID_STARTER=price_starter_monthly
STRIPE_PRICE_ID_GROWTH=price_growth_monthly
STRIPE_PRICE_ID_SCALE=price_scale_monthly

# === N8N CONFIGURATION ===
N8N_BASE_URL=http://localhost:5678
N8N_WEBHOOK_PATH=/webhook
N8N_API_KEY=your-n8n-api-key  # If authentication enabled
N8N_BASIC_AUTH_USER=admin
N8N_BASIC_AUTH_PASSWORD=admin

# === STOCK MEDIA APIs ===
PEXELS_API_KEY=your-pexels-api-key
PIXABAY_API_KEY=your-pixabay-api-key

# === COST MANAGEMENT ===
COST_PER_VIDEO_HARD_LIMIT=3.00
COST_PER_VIDEO_WARNING=2.50
COST_PER_VIDEO_TARGET=1.00
DAILY_COST_LIMIT=150.00

# === RATE LIMITING ===
RATE_LIMIT_REQUESTS_PER_MINUTE=60
RATE_LIMIT_BURST=10

# === CORS SETTINGS ===
ALLOWED_ORIGINS=http://localhost:3000,http://localhost:3001
ALLOWED_METHODS=GET,POST,PUT,PATCH,DELETE,OPTIONS
ALLOWED_HEADERS=*

# === MONITORING ===
PROMETHEUS_ENABLED=true
PROMETHEUS_PORT=9090
SENTRY_DSN=https://your-sentry-dsn@sentry.io/project-id  # Optional
SENTRY_ENVIRONMENT=development

# === FILE STORAGE ===
STORAGE_PATH=/mnt/nvme/ytempire
VIDEO_STORAGE_PATH=/mnt/nvme/ytempire/videos
THUMBNAIL_STORAGE_PATH=/mnt/nvme/ytempire/thumbnails
AUDIO_STORAGE_PATH=/mnt/nvme/ytempire/audio
TEMP_STORAGE_PATH=/mnt/nvme/ytempire/temp

# === CELERY CONFIGURATION ===
CELERY_BROKER_URL=redis://localhost:6379/1
CELERY_RESULT_BACKEND=redis://localhost:6379/2
CELERY_TASK_TIME_LIMIT=900  # 15 minutes
CELERY_TASK_SOFT_TIME_LIMIT=600  # 10 minutes

# === EMAIL CONFIGURATION (Optional) ===
SMTP_HOST=smtp.gmail.com
SMTP_PORT=587
SMTP_USER=your-email@gmail.com
SMTP_PASSWORD=your-app-password
EMAIL_FROM=noreply@ytempire.com

# === TESTING ===
TEST_DATABASE_URL=postgresql+asyncpg://ytempire:ytempire_pass@localhost:5432/ytempire_test
TEST_REDIS_URL=redis://localhost:6379/15
```

### Environment Setup Script

```bash
#!/bin/bash
# setup_environment.sh

echo "🚀 Setting up YTEMPIRE development environment..."

# Check Python version
python_version=$(python3 --version 2>&1 | grep -Po '(?<=Python )\d+\.\d+')
required_version="3.11"

if [ "$(printf '%s\n' "$required_version" "$python_version" | sort -V | head -n1)" != "$required_version" ]; then
    echo "❌ Python 3.11+ is required. Current version: $python_version"
    exit 1
fi

echo "✅ Python version check passed: $python_version"

# Create virtual environment
echo "📦 Creating virtual environment..."
python3 -m venv venv
source venv/bin/activate

# Upgrade pip
pip install --upgrade pip

# Install dependencies
echo "📚 Installing dependencies..."
pip install -r requirements.txt
pip install -r requirements-dev.txt

# Copy environment file
if [ ! -f .env ]; then
    echo "📋 Creating .env file from template..."
    cp .env.example .env
    echo "⚠️  Please update .env with your actual credentials"
fi

# Create necessary directories
echo "📁 Creating storage directories..."
mkdir -p /mnt/nvme/ytempire/{videos,thumbnails,audio,temp}

# Initialize database
echo "🗄️ Setting up PostgreSQL..."
sudo -u postgres psql <<EOF
CREATE USER ytempire WITH PASSWORD 'ytempire_pass';
CREATE DATABASE ytempire_db OWNER ytempire;
CREATE DATABASE ytempire_test OWNER ytempire;
GRANT ALL PRIVILEGES ON DATABASE ytempire_db TO ytempire;
GRANT ALL PRIVILEGES ON DATABASE ytempire_test TO ytempire;
EOF

# Run migrations
echo "🔄 Running database migrations..."
alembic upgrade head

# Start Redis
echo "🔴 Starting Redis..."
redis-server --daemonize yes

# Setup N8N
echo "🔧 Setting up N8N..."
docker run -d \
  --name n8n \
  -p 5678:5678 \
  -v n8n_data:/home/node/.n8n \
  -e N8N_BASIC_AUTH_ACTIVE=true \
  -e N8N_BASIC_AUTH_USER=admin \
  -e N8N_BASIC_AUTH_PASSWORD=admin \
  n8nio/n8n

echo "✅ Environment setup complete!"
echo "📝 Next steps:"
echo "   1. Update .env file with your API keys"
echo "   2. Run 'source venv/bin/activate' to activate virtual environment"
echo "   3. Run 'uvicorn app.main:app --reload' to start the server"
```

---

## Docker Setup

### Docker Compose Configuration

```yaml
# docker-compose.yml
version: '3.8'

services:
  # PostgreSQL Database
  postgres:
    image: postgres:14-alpine
    container_name: ytempire_postgres
    environment:
      POSTGRES_USER: ytempire
      POSTGRES_PASSWORD: ytempire_pass
      POSTGRES_DB: ytempire_db
      POSTGRES_INITDB_ARGS: "--encoding=UTF8"
    ports:
      - "5432:5432"
    volumes:
      - postgres_data:/var/lib/postgresql/data
      - ./scripts/init_db.sql:/docker-entrypoint-initdb.d/init.sql
    networks:
      - ytempire_network
    healthcheck:
      test: ["CMD-SHELL", "pg_isready -U ytempire"]
      interval: 10s
      timeout: 5s
      retries: 5

  # Redis Cache
  redis:
    image: redis:7-alpine
    container_name: ytempire_redis
    command: redis-server --appendonly yes
    ports:
      - "6379:6379"
    volumes:
      - redis_data:/data
    networks:
      - ytempire_network
    healthcheck:
      test: ["CMD", "redis-cli", "ping"]
      interval: 10s
      timeout: 5s
      retries: 5

  # N8N Workflow Engine
  n8n:
    image: n8nio/n8n:latest
    container_name: ytempire_n8n
    ports:
      - "5678:5678"
    environment:
      - N8N_BASIC_AUTH_ACTIVE=true
      - N8N_BASIC_AUTH_USER=admin
      - N8N_BASIC_AUTH_PASSWORD=admin
      - N8N_HOST=localhost
      - N8N_PORT=5678
      - N8N_PROTOCOL=http
      - DB_TYPE=postgresdb
      - DB_POSTGRESDB_HOST=postgres
      - DB_POSTGRESDB_PORT=5432
      - DB_POSTGRESDB_DATABASE=n8n
      - DB_POSTGRESDB_USER=ytempire
      - DB_POSTGRESDB_PASSWORD=ytempire_pass
    volumes:
      - n8n_data:/home/node/.n8n
    networks:
      - ytempire_network
    depends_on:
      postgres:
        condition: service_healthy

  # YTEMPIRE API (Development)
  api:
    build:
      context: .
      dockerfile: Dockerfile
    container_name: ytempire_api
    command: uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload
    ports:
      - "8000:8000"
    environment:
      - DATABASE_URL=postgresql+asyncpg://ytempire:ytempire_pass@postgres:5432/ytempire_db
      - REDIS_URL=redis://redis:6379/0
      - N8N_BASE_URL=http://n8n:5678
    volumes:
      - ./:/app
      - /mnt/nvme/ytempire:/mnt/nvme/ytempire
    networks:
      - ytempire_network
    depends_on:
      postgres:
        condition: service_healthy
      redis:
        condition: service_healthy
      n8n:
        condition: service_started

  # Celery Worker
  celery:
    build:
      context: .
      dockerfile: Dockerfile
    container_name: ytempire_celery
    command: celery -A app.celery worker --loglevel=info
    environment:
      - DATABASE_URL=postgresql+asyncpg://ytempire:ytempire_pass@postgres:5432/ytempire_db
      - REDIS_URL=redis://redis:6379/0
      - CELERY_BROKER_URL=redis://redis:6379/1
      - CELERY_RESULT_BACKEND=redis://redis:6379/2
    volumes:
      - ./:/app
      - /mnt/nvme/ytempire:/mnt/nvme/ytempire
    networks:
      - ytempire_network
    depends_on:
      - redis
      - postgres

  # Celery Beat (Scheduler)
  celery-beat:
    build:
      context: .
      dockerfile: Dockerfile
    container_name: ytempire_celery_beat
    command: celery -A app.celery beat --loglevel=info
    environment:
      - DATABASE_URL=postgresql+asyncpg://ytempire:ytempire_pass@postgres:5432/ytempire_db
      - REDIS_URL=redis://redis:6379/0
      - CELERY_BROKER_URL=redis://redis:6379/1
    volumes:
      - ./:/app
    networks:
      - ytempire_network
    depends_on:
      - redis

  # Prometheus Monitoring
  prometheus:
    image: prom/prometheus:latest
    container_name: ytempire_prometheus
    ports:
      - "9090:9090"
    volumes:
      - ./monitoring/prometheus.yml:/etc/prometheus/prometheus.yml
      - prometheus_data:/prometheus
    command:
      - '--config.file=/etc/prometheus/prometheus.yml'
      - '--storage.tsdb.path=/prometheus'
    networks:
      - ytempire_network

  # Grafana Dashboard
  grafana:
    image: grafana/grafana:latest
    container_name: ytempire_grafana
    ports:
      - "3001:3000"
    environment:
      - GF_SECURITY_ADMIN_PASSWORD=admin
      - GF_USERS_ALLOW_SIGN_UP=false
    volumes:
      - grafana_data:/var/lib/grafana
      - ./monitoring/grafana/dashboards:/etc/grafana/provisioning/dashboards
      - ./monitoring/grafana/datasources:/etc/grafana/provisioning/datasources
    networks:
      - ytempire_network
    depends_on:
      - prometheus

volumes:
  postgres_data:
  redis_data:
  n8n_data:
  prometheus_data:
  grafana_data:

networks:
  ytempire_network:
    driver: bridge
```

### Dockerfile

```dockerfile
# Dockerfile
FROM python:3.11-slim

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    gcc \
    postgresql-client \
    ffmpeg \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

# Create non-root user
RUN useradd -m ytempire && chown -R ytempire:ytempire /app
USER ytempire

# Expose port
EXPOSE 8000

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
  CMD curl -f http://localhost:8000/health || exit 1

# Default command
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
```

---

## Database Initialization

### Database Setup Script

```sql
-- scripts/init_db.sql
-- PostgreSQL initialization script for YTEMPIRE

-- Create extensions
CREATE EXTENSION IF NOT EXISTS "uuid-ossp";
CREATE EXTENSION IF NOT EXISTS "pg_trgm";
CREATE EXTENSION IF NOT EXISTS "btree_gin";

-- Create enum types
CREATE TYPE user_role AS ENUM ('user', 'admin');
CREATE TYPE user_tier AS ENUM ('free', 'starter', 'growth', 'scale');
CREATE TYPE channel_status AS ENUM ('pending_oauth', 'active', 'paused', 'error');
CREATE TYPE video_status AS ENUM ('queued', 'processing', 'completed', 'published', 'failed');

-- Create database for N8N
CREATE DATABASE n8n;

-- Grant permissions
GRANT ALL PRIVILEGES ON DATABASE ytempire_db TO ytempire;
GRANT ALL PRIVILEGES ON DATABASE ytempire_test TO ytempire;
GRANT ALL PRIVILEGES ON DATABASE n8n TO ytempire;
```

### Alembic Configuration

```python
# alembic.ini
[alembic]
script_location = migrations
prepend_sys_path = .
version_path_separator = os
sqlalchemy.url = postgresql+asyncpg://ytempire:ytempire_pass@localhost:5432/ytempire_db

[post_write_hooks]
hooks = black
black.type = console_scripts
black.entrypoint = black
black.options = -l 88

[loggers]
keys = root,sqlalchemy,alembic

[handlers]
keys = console

[formatters]
keys = generic

[logger_root]
level = WARN
handlers = console
qualname =

[logger_sqlalchemy]
level = WARN
handlers =
qualname = sqlalchemy.engine

[logger_alembic]
level = INFO
handlers =
qualname = alembic

[handler_console]
class = StreamHandler
args = (sys.stderr,)
level = NOTSET
formatter = generic

[formatter_generic]
format = %(levelname)-5.5s [%(name)s] %(message)s
datefmt = %H:%M:%S
```

### Initial Migration

```python
# migrations/env.py
from logging.config import fileConfig
from sqlalchemy import engine_from_config
from sqlalchemy import pool
from alembic import context
import asyncio
from sqlalchemy.ext.asyncio import AsyncEngine

from app.database import Base
from app.models import *  # Import all models

config = context.config
fileConfig(config.config_file_name)
target_metadata = Base.metadata

def run_migrations_offline():
    """Run migrations in 'offline' mode."""
    url = config.get_main_option("sqlalchemy.url")
    context.configure(
        url=url,
        target_metadata=target_metadata,
        literal_binds=True,
        dialect_opts={"paramstyle": "named"},
    )

    with context.begin_transaction():
        context.run_migrations()

def do_run_migrations(connection):
    context.configure(connection=connection, target_metadata=target_metadata)
    
    with context.begin_transaction():
        context.run_migrations()

async def run_migrations_online():
    """Run migrations in 'online' mode."""
    configuration = config.get_section(config.config_ini_section)
    configuration["sqlalchemy.url"] = config.get_main_option("sqlalchemy.url")
    
    connectable = AsyncEngine(
        engine_from_config(
            configuration,
            prefix="sqlalchemy.",
            poolclass=pool.NullPool,
            future=True,
        )
    )

    async with connectable.connect() as connection:
        await connection.run_sync(do_run_migrations)

if context.is_offline_mode():
    run_migrations_offline()
else:
    asyncio.run(run_migrations_online())
```

---

## Local Development Setup

### Python Requirements

```txt
# requirements.txt
# Core Framework
fastapi==0.104.1
uvicorn[standard]==0.24.0.post1
pydantic==2.5.2
pydantic-settings==2.1.0

# Database
sqlalchemy==2.0.23
alembic==1.13.0
asyncpg==0.29.0
psycopg2-binary==2.9.9

# Redis
redis==5.0.1
hiredis==2.2.3

# Authentication
python-jose[cryptography]==3.3.0
passlib[bcrypt]==1.7.4
python-multipart==0.0.6

# YouTube API
google-api-python-client==2.111.0
google-auth==2.25.2
google-auth-oauthlib==1.1.0
google-auth-httplib2==0.1.1

# OpenAI
openai==1.6.1

# Text-to-Speech
google-cloud-texttospeech==2.15.0

# Stripe
stripe==7.8.0

# HTTP Clients
httpx==0.25.2
aiohttp==3.9.1

# Task Queue
celery==5.3.4
flower==2.0.1

# Monitoring
prometheus-client==0.19.0
sentry-sdk[fastapi]==1.39.1

# Utilities
python-dotenv==1.0.0
pyyaml==6.0.1
click==8.1.7

# Development Tools
pytest==7.4.3
pytest-asyncio==0.21.1
pytest-cov==4.1.0
black==23.12.0
flake8==6.1.0
mypy==1.7.1
ipython==8.18.1
```

### Development Tools Setup

```bash
# Install pre-commit hooks
pip install pre-commit
pre-commit install

# .pre-commit-config.yaml
repos:
  - repo: https://github.com/ambv/black
    rev: 23.12.0
    hooks:
      - id: black
        language_version: python3.11
        
  - repo: https://github.com/pycqa/flake8
    rev: 6.1.0
    hooks:
      - id: flake8
        args: ['--max-line-length=88', '--extend-ignore=E203']
        
  - repo: https://github.com/pre-commit/mirrors-mypy
    rev: v1.7.1
    hooks:
      - id: mypy
        additional_dependencies: [types-all]
```

### VS Code Configuration

```json
// .vscode/settings.json
{
    "python.defaultInterpreter": "${workspaceFolder}/venv/bin/python",
    "python.linting.enabled": true,
    "python.linting.pylintEnabled": false,
    "python.linting.flake8Enabled": true,
    "python.linting.flake8Args": [
        "--max-line-length=88",
        "--extend-ignore=E203"
    ],
    "python.formatting.provider": "black",
    "python.formatting.blackArgs": [
        "--line-length=88"
    ],
    "editor.formatOnSave": true,
    "editor.codeActionsOnSave": {
        "source.organizeImports": true
    },
    "python.testing.pytestEnabled": true,
    "python.testing.pytestArgs": [
        "tests"
    ],
    "[python]": {
        "editor.rulers": [88]
    }
}
```

```json
// .vscode/launch.json
{
    "version": "0.2.0",
    "configurations": [
        {
            "name": "FastAPI",
            "type": "python",
            "request": "launch",
            "module": "uvicorn",
            "args": [
                "app.main:app",
                "--reload",
                "--host",
                "0.0.0.0",
                "--port",
                "8000"
            ],
            "jinja": true,
            "justMyCode": false,
            "env": {
                "PYTHONPATH": "${workspaceFolder}"
            }
        },
        {
            "name": "Celery Worker",
            "type": "python",
            "request": "launch",
            "module": "celery",
            "args": [
                "-A",
                "app.celery",
                "worker",
                "--loglevel=info"
            ]
        },
        {
            "name": "Pytest",
            "type": "python",
            "request": "launch",
            "module": "pytest",
            "args": [
                "-v",
                "--cov=app",
                "--cov-report=html"
            ]
        }
    ]
}
```

---

## External Services Setup

### YouTube API Setup

1. **Create Google Cloud Project**
   ```
   1. Go to https://console.cloud.google.com
   2. Create new project: "ytempire-production"
   3. Enable YouTube Data API v3
   4. Create OAuth 2.0 credentials
   5. Add redirect URI: http://localhost:8000/api/v1/auth/youtube/callback
   6. Download credentials as JSON
   7. Set up 15 Google accounts for video distribution
   ```

2. **Configure Service Account**
   ```bash
   # Create service account for non-OAuth operations
   gcloud iam service-accounts create ytempire-service \
     --display-name="YTEMPIRE Service Account"
   
   # Grant necessary permissions
   gcloud projects add-iam-policy-binding ytempire-production \
     --member="serviceAccount:ytempire-service@ytempire-production.iam.gserviceaccount.com" \
     --role="roles/youtube.admin"
   
   # Download service account key
   gcloud iam service-accounts keys create ~/ytempire-service-key.json \
     --iam-account=ytempire-service@ytempire-production.iam.gserviceaccount.com
   ```

### OpenAI API Setup

1. **Get API Key**
   ```
   1. Go to https://platform.openai.com
   2. Navigate to API Keys
   3. Create new secret key
   4. Set usage limits: $50/day
   5. Enable GPT-3.5 and GPT-4 models
   ```

2. **Configure Rate Limits**
   ```python
   # OpenAI rate limit configuration
   OPENAI_RATE_LIMITS = {
       "gpt-3.5-turbo": {
           "rpm": 3500,  # Requests per minute
           "tpm": 90000,  # Tokens per minute
           "dpf": 200     # Dollars per day
       },
       "gpt-4": {
           "rpm": 500,
           "tpm": 10000,
           "dpf": 100
       }
   }
   ```

### Stripe Setup

1. **Test Mode Configuration**
   ```
   1. Go to https://dashboard.stripe.com
   2. Switch to Test Mode
   3. Get test API keys
   4. Create webhook endpoint
   5. Create test products and prices
   ```

2. **Webhook Configuration**
   ```bash
   # Install Stripe CLI for testing
   brew install stripe/stripe-cli/stripe
   
   # Login to Stripe
   stripe login
   
   # Forward webhooks to local
   stripe listen --forward-to localhost:8000/api/v1/webhooks/stripe
   
   # Trigger test events
   stripe trigger payment_intent.succeeded
   ```

### N8N Workflow Setup

1. **Import Base Workflows**
   ```json
   // workflows/video_generation.json
   {
     "name": "Video Generation Pipeline",
     "nodes": [
       {
         "name": "Start",
         "type": "n8n-nodes-base.webhook",
         "position": [250, 300],
         "webhookId": "video-generation",
         "parameters": {
           "httpMethod": "POST",
           "path": "video-generate"
         }
       },
       {
         "name": "Check Cost",
         "type": "n8n-nodes-base.httpRequest",
         "position": [450, 300],
         "parameters": {
           "url": "http://api:8000/api/v1/n8n/track-cost",
           "method": "POST",
           "bodyParametersJson": {
             "video_id": "={{$json[\"video_id\"]}}",
             "service": "estimate",
             "amount": 0
           }
         }
       },
       {
         "name": "Generate Script",
         "type": "n8n-nodes-base.openAi",
         "position": [650, 300],
         "parameters": {
           "operation": "completion",
           "model": "gpt-3.5-turbo",
           "prompt": "={{$json[\"prompt\"]}}"
         }
       }
     ]
   }
   ```

2. **Custom Node Development**
   ```javascript
   // n8n-nodes-ytempire/nodes/CostTracker.node.ts
   import {
     IExecuteFunctions,
     INodeExecutionData,
     INodeType,
     INodeTypeDescription,
   } from 'n8n-workflow';
   
   export class CostTracker implements INodeType {
     description: INodeTypeDescription = {
       displayName: 'YTEMPIRE Cost Tracker',
       name: 'ytempireCostTracker',
       group: ['transform'],
       version: 1,
       description: 'Track costs for video generation',
       defaults: {
         name: 'Cost Tracker',
         color: '#00B900',
       },
       inputs: ['main'],
       outputs: ['main'],
       properties: [
         {
           displayName: 'Video ID',
           name: 'videoId',
           type: 'string',
           required: true,
           default: '',
         },
         {
           displayName: 'Service',
           name: 'service',
           type: 'string',
           required: true,
           default: '',
         },
         {
           displayName: 'Amount',
           name: 'amount',
           type: 'number',
           required: true,
           default: 0,
         },
       ],
     };
   
     async execute(this: IExecuteFunctions): Promise<INodeExecutionData[][]> {
       const items = this.getInputData();
       const returnData: INodeExecutionData[] = [];
   
       for (let i = 0; i < items.length; i++) {
         const videoId = this.getNodeParameter('videoId', i) as string;
         const service = this.getNodeParameter('service', i) as string;
         const amount = this.getNodeParameter('amount', i) as number;
   
         // Call API to track cost
         const response = await this.helpers.httpRequest({
           method: 'POST',
           url: 'http://api:8000/api/v1/n8n/track-cost',
           body: {
             video_id: videoId,
             service: service,
             amount: amount,
           },
         });
   
         returnData.push({
           json: response,
         });
       }
   
       return [returnData];
     }
   }
   ```

---

## Testing Environment

### Test Database Setup

```bash
#!/bin/bash
# scripts/setup_test_db.sh

echo "Setting up test database..."

# Create test database
sudo -u postgres psql <<EOF
CREATE DATABASE ytempire_test OWNER ytempire;
GRANT ALL PRIVILEGES ON DATABASE ytempire_test TO ytempire;
EOF

# Run migrations on test database
DATABASE_URL=postgresql+asyncpg://ytempire:ytempire_pass@localhost:5432/ytempire_test \
  alembic upgrade head

echo "Test database ready!"
```

### Test Data Fixtures

```python
# tests/fixtures.py
import pytest
from typing import AsyncGenerator
from sqlalchemy.ext.asyncio import AsyncSession
from app.models import User, Channel, Video
from app.core.security import get_password_hash

@pytest.fixture
async def test_user(db_session: AsyncSession) -> User:
    """Create test user"""
    user = User(
        email="test@example.com",
        password_hash=get_password_hash("testpass123"),
        full_name="Test User",
        tier="starter",
        channel_limit=5
    )
    db_session.add(user)
    await db_session.commit()
    await db_session.refresh(user)
    return user

@pytest.fixture
async def test_channel(db_session: AsyncSession, test_user: User) -> Channel:
    """Create test channel"""
    channel = Channel(
        user_id=test_user.id,
        name="Test Channel",
        niche="education",
        status="active",
        youtube_channel_id="UCtest123"
    )
    db_session.add(channel)
    await db_session.commit()
    await db_session.refresh(channel)
    return channel

@pytest.fixture
async def test_video(db_session: AsyncSession, test_channel: Channel) -> Video:
    """Create test video"""
    video = Video(
        channel_id=test_channel.id,
        title="Test Video",
        status="completed",
        cost=2.35,
        generation_time_seconds=450
    )
    db_session.add(video)
    await db_session.commit()
    await db_session.refresh(video)
    return video

# Sample test data
SAMPLE_VIDEO_TOPICS = [
    "10 Python Tips for Beginners",
    "How to Start a YouTube Channel",
    "Best Productivity Apps 2025",
    "Understanding Machine Learning",
    "Healthy Meal Prep Ideas"
]

SAMPLE_NICHES = [
    "education",
    "technology",
    "lifestyle",
    "gaming",
    "finance"
]

TEST_YOUTUBE_ACCOUNTS = [
    {
        "email": f"ytempire.test{i}@gmail.com",
        "channel_id": f"UCtest{i:03d}",
        "is_reserve": i > 12
    }
    for i in range(1, 16)
]
```

### Integration Test Setup

```python
# tests/integration/test_full_pipeline.py
import pytest
import asyncio
from httpx import AsyncClient

@pytest.mark.asyncio
async def test_full_video_generation_pipeline(
    client: AsyncClient,
    auth_headers: dict,
    test_channel: Channel
):
    """Test complete video generation pipeline"""
    
    # Step 1: Generate video
    response = await client.post(
        "/api/v1/videos/generate",
        json={
            "channel_id": str(test_channel.id),
            "topic": "Test Video Topic",
            "style": "educational",
            "length_minutes": 8,
            "optimization_level": "economy"
        },
        headers=auth_headers
    )
    
    assert response.status_code == 202
    data = response.json()
    video_id = data["video_id"]
    job_id = data["job_id"]
    
    # Step 2: Check status
    for _ in range(10):  # Poll for 10 seconds
        response = await client.get(
            f"/api/v1/videos/{video_id}/status",
            headers=auth_headers
        )
        
        status_data = response.json()
        if status_data["status"] == "completed":
            break
            
        await asyncio.sleep(1)
    
    # Step 3: Verify completion
    response = await client.get(
        f"/api/v1/videos/{video_id}",
        headers=auth_headers
    )
    
    assert response.status_code == 200
    video_data = response.json()
    assert video_data["status"] == "completed"
    assert video_data["cost_breakdown"]["total"] <= 3.00
    
    # Step 4: Check cost tracking
    response = await client.get(
        "/api/v1/costs/current",
        headers=auth_headers
    )
    
    assert response.status_code == 200
    cost_data = response.json()
    assert cost_data["daily"]["videos_generated"] >= 1
```

### Load Testing

```python
# tests/load/locustfile.py
from locust import HttpUser, task, between
import json

class YTEmpireUser(HttpUser):
    wait_time = between(1, 3)
    
    def on_start(self):
        """Login and get token"""
        response = self.client.post(
            "/api/v1/auth/login",
            data={
                "username": "loadtest@example.com",
                "password": "loadtest123"
            }
        )
        self.token = response.json()["access_token"]
        self.headers = {"Authorization": f"Bearer {self.token}"}
    
    @task(3)
    def list_channels(self):
        """List channels endpoint"""
        self.client.get("/api/v1/channels", headers=self.headers)
    
    @task(2)
    def get_analytics(self):
        """Get analytics summary"""
        self.client.get("/api/v1/analytics/summary", headers=self.headers)
    
    @task(1)
    def generate_video(self):
        """Generate video (less frequent)"""
        self.client.post(
            "/api/v1/videos/generate",
            json={
                "channel_id": "test-channel-id",
                "topic": "Load Test Video",
                "style": "educational",
                "optimization_level": "economy"
            },
            headers=self.headers
        )

# Run with: locust -f locustfile.py --host=http://localhost:8000
```

---

## Monitoring Setup

### Prometheus Configuration

```yaml
# monitoring/prometheus.yml
global:
  scrape_interval: 15s
  evaluation_interval: 15s

scrape_configs:
  - job_name: 'ytempire-api'
    static_configs:
      - targets: ['api:8000']
    metrics_path: '/metrics'
    
  - job_name: 'postgres'
    static_configs:
      - targets: ['postgres-exporter:9187']
      
  - job_name: 'redis'
    static_configs:
      - targets: ['redis-exporter:9121']
      
  - job_name: 'node'
    static_configs:
      - targets: ['node-exporter:9100']
```

### Grafana Dashboard

```json
// monitoring/grafana/dashboards/api-dashboard.json
{
  "dashboard": {
    "title": "YTEMPIRE API Dashboard",
    "panels": [
      {
        "title": "Request Rate",
        "targets": [
          {
            "expr": "rate(http_requests_total[5m])",
            "legendFormat": "{{method}} {{endpoint}}"
          }
        ]
      },
      {
        "title": "Response Time P95",
        "targets": [
          {
            "expr": "histogram_quantile(0.95, rate(http_request_duration_seconds_bucket[5m]))",
            "legendFormat": "p95"
          }
        ]
      },
      {
        "title": "Cost per Video",
        "targets": [
          {
            "expr": "avg(video_cost_dollars)",
            "legendFormat": "Average Cost"
          }
        ]
      },
      {
        "title": "Videos Generated",
        "targets": [
          {
            "expr": "rate(videos_generated_total[1h])",
            "legendFormat": "Videos/hour"
          }
        ]
      }
    ]
  }
}
```

---

## Quick Start Commands

```bash
# Complete setup in one command
make setup

# Start all services
make up

# Run tests
make test

# View logs
make logs

# Stop all services
make down

# Clean everything
make clean
```

### Makefile

```makefile
# Makefile
.PHONY: help setup up down logs test clean

help:
	@echo "Available commands:"
	@echo "  make setup  - Set up development environment"
	@echo "  make up     - Start all services"
	@echo "  make down   - Stop all services"
	@echo "  make logs   - View logs"
	@echo "  make test   - Run tests"
	@echo "  make clean  - Clean up everything"

setup:
	./scripts/setup_environment.sh
	docker-compose build
	alembic upgrade head
	./scripts/setup_test_db.sh

up:
	docker-compose up -d
	@echo "Services started!"
	@echo "API: http://localhost:8000"
	@echo "N8N: http://localhost:5678"
	@echo "Grafana: http://localhost:3001"

down:
	docker-compose down

logs:
	docker-compose logs -f

test:
	pytest tests/ -v --cov=app --cov-report=html

clean:
	docker-compose down -v
	rm -rf venv/
	rm -rf __pycache__/
	rm -rf .pytest_cache/
	rm -rf .coverage
	rm -rf htmlcov/
	find . -type d -name "__pycache__" -exec rm -rf {} +
```

---

This completes the comprehensive Development Environment Setup Guide with all necessary configurations, scripts, and tools for the API Development Engineer to successfully set up and run the YTEMPIRE backend system.