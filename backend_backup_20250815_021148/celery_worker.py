#!/usr/bin/env python
"""
Celery Worker Startup Script
Run with: celery -A celery_worker worker --loglevel=info
"""
import os
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

from app.core.celery_app import celery_app

if __name__ == '__main__':
    celery_app.start()