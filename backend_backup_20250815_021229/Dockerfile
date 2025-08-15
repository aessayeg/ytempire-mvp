# Multi-stage Dockerfile for Backend
# Stage 1: Base dependencies
FROM python:3.11-slim as base

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    postgresql-client \
    curl \
    wget \
    git \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip \
    && pip install --no-cache-dir -r requirements.txt

# Stage 2: Development
FROM base as development

# Install development dependencies
RUN apt-get update && apt-get install -y \
    vim \
    htop \
    net-tools \
    && rm -rf /var/lib/apt/lists/*

# Install development Python packages
RUN pip install --no-cache-dir \
    pytest \
    pytest-cov \
    pytest-asyncio \
    black \
    flake8 \
    mypy \
    ipython \
    debugpy

# Copy application code
COPY . .

# Create necessary directories
RUN mkdir -p /app/uploads /app/logs /app/data

# Set environment for development
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1

# Expose port
EXPOSE 8000

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# Development command with hot reload
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000", "--reload"]

# Stage 3: Testing
FROM base as testing

# Install testing dependencies
RUN pip install --no-cache-dir \
    pytest \
    pytest-cov \
    pytest-asyncio \
    pytest-mock \
    pytest-xdist \
    httpx \
    faker

# Copy application code
COPY . .

# Run tests
RUN pytest tests/ -v --cov=app --cov-report=term-missing || true

# Stage 4: Production build
FROM base as builder

# Copy application code
COPY . .

# Compile Python files for optimization
RUN python -m compileall -b .

# Remove unnecessary files
RUN find . -name "*.py" -delete \
    && find . -name "__pycache__" -type d -exec rm -rf {} + 2>/dev/null || true \
    && find . -name "*.pyc" -delete \
    && rm -rf tests/ .pytest_cache/ .coverage

# Stage 5: Production
FROM python:3.11-slim as production

# Create non-root user
RUN groupadd -r ytempire && useradd -r -g ytempire ytempire

# Set working directory
WORKDIR /app

# Install only runtime dependencies
RUN apt-get update && apt-get install -y \
    postgresql-client \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy Python packages from builder
COPY --from=builder /usr/local/lib/python3.11/site-packages /usr/local/lib/python3.11/site-packages
COPY --from=builder /usr/local/bin /usr/local/bin

# Copy compiled application from builder
COPY --from=builder --chown=ytempire:ytempire /app /app

# Create necessary directories with proper permissions
RUN mkdir -p /app/uploads /app/logs /app/data \
    && chown -R ytempire:ytempire /app

# Switch to non-root user
USER ytempire

# Set production environment variables
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONOPTIMIZE=2

# Expose port
EXPOSE 8000

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=40s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# Production command with gunicorn
CMD ["gunicorn", "app.main:app", "-w", "4", "-k", "uvicorn.workers.UvicornWorker", "--bind", "0.0.0.0:8000", "--access-logfile", "-", "--error-logfile", "-"]

# Stage 6: GPU-enabled production (for video processing)
FROM nvidia/cuda:12.2.0-runtime-ubuntu22.04 as gpu

# Install Python and dependencies
RUN apt-get update && apt-get install -y \
    python3.11 \
    python3-pip \
    ffmpeg \
    libgl1-mesa-glx \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libgomp1 \
    wget \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Set Python 3.11 as default
RUN update-alternatives --install /usr/bin/python python /usr/bin/python3.11 1 \
    && update-alternatives --install /usr/bin/pip pip /usr/bin/pip3 1

# Set working directory
WORKDIR /app

# Copy requirements and install
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip \
    && pip install --no-cache-dir -r requirements.txt \
    && pip install --no-cache-dir \
    opencv-python-headless \
    torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

# Copy application
COPY --chown=1000:1000 . .

# Create directories
RUN mkdir -p /app/uploads /app/logs /app/data /app/videos

# Set environment
ENV PYTHONUNBUFFERED=1
ENV CUDA_VISIBLE_DEVICES=0
ENV OPENCV_CUDA_ENABLE=1

# Expose port
EXPOSE 8000

# GPU health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD python -c "import torch; assert torch.cuda.is_available()" && curl -f http://localhost:8000/health || exit 1

# Command for GPU workers
CMD ["celery", "-A", "app.core.celery_app", "worker", "--loglevel=info", "--concurrency=2", "-Q", "video_processing,gpu_tasks"]