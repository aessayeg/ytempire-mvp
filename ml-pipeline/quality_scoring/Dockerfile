# Quality Scoring Service Dockerfile
FROM python:3.11-slim

# Set environment variables
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1

# Install system dependencies
RUN apt-get update && apt-get install -y \
    ffmpeg \
    libsndfile1 \
    git \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Create working directory
WORKDIR /app

# Copy requirements and install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip
RUN pip install --no-cache-dir -r requirements.txt

# Download NLTK data
RUN python -c "import nltk; nltk.download('punkt'); nltk.download('averaged_perceptron_tagger')"

# Copy application code
COPY . .

# Create necessary directories
RUN mkdir -p /app/data /app/logs /app/models /app/temp

# Set permissions
RUN chmod +x quality_scorer.py quality_api.py

# Expose port
EXPOSE 8001

# Health check
HEALTHCHECK --interval=60s --timeout=30s --start-period=10s --retries=3 \
    CMD curl -f http://localhost:8001/health || exit 1

# Default command
CMD ["python", "quality_api.py"]