"""
Core Configuration
Owner: API Developer
"""

from typing import List, Optional
from pydantic_settings import BaseSettings
from pydantic import AnyHttpUrl, validator
import os


class Settings(BaseSettings):
    # Project Info
    PROJECT_NAME: str = "YTEmpire"
    VERSION: str = "0.1.0"
    API_V1_STR: str = "/api/v1"
    DEBUG: bool = True
    
    # Security
    SECRET_KEY: str = os.getenv("SECRET_KEY", "your-secret-key-change-in-production")
    ALGORITHM: str = "HS256"
    ACCESS_TOKEN_EXPIRE_MINUTES: int = 30
    REFRESH_TOKEN_EXPIRE_DAYS: int = 7
    
    # Database
    POSTGRES_SERVER: str = os.getenv("POSTGRES_SERVER", "localhost")
    POSTGRES_USER: str = os.getenv("POSTGRES_USER", "ytempire")
    POSTGRES_PASSWORD: str = os.getenv("POSTGRES_PASSWORD", "ytempire_pass")
    POSTGRES_DB: str = os.getenv("POSTGRES_DB", "ytempire_db")
    DATABASE_URL: Optional[str] = None
    
    @validator("DATABASE_URL", pre=True)
    def assemble_db_connection(cls, v: Optional[str], values: dict) -> str:
        if isinstance(v, str):
            return v
        return f"postgresql+asyncpg://{values.get('POSTGRES_USER')}:{values.get('POSTGRES_PASSWORD')}@{values.get('POSTGRES_SERVER')}/{values.get('POSTGRES_DB')}"
    
    # Redis
    REDIS_HOST: str = os.getenv("REDIS_HOST", "localhost")
    REDIS_PORT: int = 6379
    REDIS_DB: int = 0
    REDIS_URL: str = f"redis://{REDIS_HOST}:{REDIS_PORT}/{REDIS_DB}"
    
    # CORS
    BACKEND_CORS_ORIGINS: List[AnyHttpUrl] = [
        "http://localhost:3000",
        "http://localhost:5173",
        "http://localhost:8000"
    ]
    
    # AI Service Configuration (VP of AI responsibility)
    OPENAI_API_KEY: str = os.getenv("OPENAI_API_KEY", "")
    OPENAI_ORG_ID: str = os.getenv("OPENAI_ORG_ID", "")
    ELEVENLABS_API_KEY: str = os.getenv("ELEVENLABS_API_KEY", "")
    GOOGLE_CLOUD_TTS_KEY: str = os.getenv("GOOGLE_CLOUD_TTS_KEY", "")
    
    # Cost Limits (VP of AI requirement: <$3/video)
    MAX_COST_PER_VIDEO: float = 3.0
    COST_WARNING_THRESHOLD: float = 2.5
    
    # YouTube API (Integration Specialist)
    YOUTUBE_API_KEY: str = os.getenv("YOUTUBE_API_KEY", "")
    YOUTUBE_CLIENT_ID: str = os.getenv("YOUTUBE_CLIENT_ID", "")
    YOUTUBE_CLIENT_SECRET: str = os.getenv("YOUTUBE_CLIENT_SECRET", "")
    YOUTUBE_QUOTA_LIMIT: int = 10000  # Daily quota
    
    # Channel Limits (Product Owner requirement)
    MAX_CHANNELS_PER_USER: int = 5
    MAX_VIDEOS_PER_DAY: int = 10
    
    # Celery Configuration (Data Pipeline Engineer)
    CELERY_BROKER_URL: str = f"redis://{REDIS_HOST}:{REDIS_PORT}/1"
    CELERY_RESULT_BACKEND: str = f"redis://{REDIS_HOST}:{REDIS_PORT}/2"
    
    # N8N Webhook (Integration Specialist)
    N8N_WEBHOOK_URL: str = os.getenv("N8N_WEBHOOK_URL", "http://localhost:5678/webhook")
    
    # Monitoring (Platform Ops Lead)
    PROMETHEUS_ENABLED: bool = True
    GRAFANA_URL: str = "http://localhost:3001"
    
    class Config:
        case_sensitive = True
        env_file = ".env"


settings = Settings()