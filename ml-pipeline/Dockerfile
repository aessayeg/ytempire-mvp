# Multi-stage Dockerfile for ML Pipeline
# Stage 1: Base dependencies
FROM python:3.11-slim as base

# Set environment variables
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    DEBIAN_FRONTEND=noninteractive

# Set working directory
WORKDIR /app

# Install system dependencies required for ML pipeline
RUN apt-get update && apt-get install -y \
    # Build tools
    gcc \
    g++ \
    make \
    cmake \
    # Audio/Video processing
    ffmpeg \
    libsndfile1 \
    libsndfile1-dev \
    # OpenCV dependencies
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libgomp1 \
    libglib2.0-0 \
    # Git for some pip packages
    git \
    # Network tools
    curl \
    wget \
    # Clean up
    && rm -rf /var/lib/apt/lists/*

# Stage 2: Python dependencies
FROM base as dependencies

# Copy requirements file
COPY requirements.txt .

# Install Python dependencies
RUN pip install --upgrade pip setuptools wheel && \
    pip install --no-cache-dir -r requirements.txt

# Download spacy language model (if needed)
RUN python -m spacy download en_core_web_sm || true

# Stage 3: Application
FROM dependencies as application

# Copy application code
COPY . /app/

# Create necessary directories
RUN mkdir -p /app/logs \
    /app/models \
    /app/data \
    /app/outputs \
    /app/cache

# Set Python path
ENV PYTHONPATH=/app:$PYTHONPATH

# Create non-root user for security
RUN useradd -m -u 1000 mlpipeline && \
    chown -R mlpipeline:mlpipeline /app

# Switch to non-root user
USER mlpipeline

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD python -c "import torch; import transformers; print('ML Pipeline healthy')" || exit 1

# Default command (can be overridden)
CMD ["python", "-m", "services.trend_detection"]

# Expose port if needed for API services
EXPOSE 8001

# Labels for metadata
LABEL maintainer="YTEmpire Team" \
      version="1.0.0" \
      description="ML Pipeline for YTEmpire - Video Generation and AI Services"