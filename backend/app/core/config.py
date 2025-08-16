"""
YTEmpire Configuration Settings
"""
from typing import List, Union, Optional, Dict, Any
from pydantic_settings import BaseSettings
from pydantic import AnyHttpUrl, validator, EmailStr
import secrets
import os
from pathlib import Path
from dotenv import load_dotenv

# Load .env file
load_dotenv()


class Settings(BaseSettings):
    # Project Info
    PROJECT_NAME: str = "YTEmpire MVP"
    VERSION: str = "0.1.0"
    API_V1_STR: str = "/api/v1"
    ENVIRONMENT: str = "development"

    # Security
    SECRET_KEY: str = secrets.token_urlsafe(32)
    ACCESS_TOKEN_EXPIRE_MINUTES: int = 60 * 24 * 7  # 7 days
    REFRESH_TOKEN_EXPIRE_MINUTES: int = 60 * 24 * 30  # 30 days
    ALGORITHM: str = "HS256"

    # CORS
    BACKEND_CORS_ORIGINS: List[AnyHttpUrl] = [
        "http://localhost:3000",
        "http://localhost:5173",
        "http://127.0.0.1:3000",
        "http://127.0.0.1:5173",
    ]
    ALLOWED_HOSTS: List[str] = ["*"]

    # Database
    POSTGRES_USER: str = "ytempire"
    POSTGRES_PASSWORD: str = "admin"
    POSTGRES_HOST: str = "localhost"
    POSTGRES_PORT: str = "5432"
    POSTGRES_DB: str = "ytempire_db"
    DATABASE_URL: Optional[str] = None

    # Database Pool Configuration (Exposed for production tuning)
    DATABASE_POOL_SIZE: int = 50  # Base pool size
    DATABASE_MAX_OVERFLOW: int = 150  # Maximum overflow connections
    DATABASE_POOL_TIMEOUT: int = 30  # Timeout for getting connection from pool
    DATABASE_POOL_RECYCLE: int = 1800  # Recycle connections after 30 minutes
    DATABASE_ECHO: bool = False  # Enable SQL logging (debug only)

    @validator("DATABASE_URL", pre=True)
    def assemble_db_connection(cls, v: Optional[str], values: Dict[str, Any]) -> Any:
        if isinstance(v, str):
            return v
        return f"postgresql+asyncpg://{values.get('POSTGRES_USER')}:{values.get('POSTGRES_PASSWORD')}@{values.get('POSTGRES_HOST')}:{values.get('POSTGRES_PORT')}/{values.get('POSTGRES_DB')}"

    # Redis
    REDIS_HOST: str = "localhost"
    REDIS_PORT: int = 6379
    REDIS_DB: int = 0
    REDIS_PASSWORD: Optional[str] = None
    REDIS_URL: Optional[str] = None

    @validator("REDIS_URL", pre=True)
    def assemble_redis_connection(cls, v: Optional[str], values: Dict[str, Any]) -> Any:
        if isinstance(v, str):
            return v
        password = (
            f":{values.get('REDIS_PASSWORD')}@" if values.get("REDIS_PASSWORD") else ""
        )
        return f"redis://{password}{values.get('REDIS_HOST')}:{values.get('REDIS_PORT')}/{values.get('REDIS_DB')}"

    # Celery
    CELERY_BROKER_URL: Optional[str] = None
    CELERY_RESULT_BACKEND: Optional[str] = None

    @validator("CELERY_BROKER_URL", pre=True)
    def set_celery_broker(cls, v: Optional[str], values: Dict[str, Any]) -> Any:
        if isinstance(v, str):
            return v
        return values.get("REDIS_URL")

    @validator("CELERY_RESULT_BACKEND", pre=True)
    def set_celery_backend(cls, v: Optional[str], values: Dict[str, Any]) -> Any:
        if isinstance(v, str):
            return v
        return values.get("REDIS_URL")

    # AI Services
    OPENAI_API_KEY: Optional[str] = None
    OPENAI_ORG_ID: Optional[str] = None
    ANTHROPIC_API_KEY: Optional[str] = None
    ELEVENLABS_API_KEY: Optional[str] = None
    GOOGLE_CLOUD_PROJECT: Optional[str] = None
    GOOGLE_APPLICATION_CREDENTIALS: Optional[str] = None

    # YouTube API
    YOUTUBE_API_KEY: Optional[str] = None  # Single key for MVP
    YOUTUBE_API_KEYS: List[str] = []
    YOUTUBE_CLIENT_ID: Optional[str] = None
    YOUTUBE_CLIENT_SECRET: Optional[str] = None
    YOUTUBE_REDIRECT_URI: str = "http://localhost:8000/api/v1/auth/youtube/callback"

    # Stock Footage APIs
    PEXELS_API_KEY: Optional[str] = None
    PIXABAY_API_KEY: Optional[str] = None

    # AI Service APIs
    OPENAI_API_KEY: Optional[str] = None
    ELEVENLABS_API_KEY: Optional[str] = None
    ANTHROPIC_API_KEY: Optional[str] = None
    GOOGLE_TTS_API_KEY: Optional[str] = None

    # Cost Optimization
    MAX_COST_PER_VIDEO: float = 3.00  # $3 per video target
    DAILY_API_BUDGET: float = 100.00  # $100 daily budget
    MONTHLY_API_BUDGET: float = 10000.00  # $10K monthly budget

    # Performance Targets
    API_RESPONSE_TIME_TARGET: float = 0.5  # 500ms
    VIDEO_GENERATION_TIME_TARGET: int = 600  # 10 minutes
    SYSTEM_UPTIME_TARGET: float = 0.999  # 99.9%

    # Rate Limiting
    RATE_LIMIT_PER_MINUTE: int = 60
    RATE_LIMIT_PER_HOUR: int = 1000

    # File Storage
    UPLOAD_DIR: Path = Path("uploads")
    MAX_UPLOAD_SIZE: int = 100 * 1024 * 1024  # 100MB
    ALLOWED_EXTENSIONS: List[str] = [".mp4", ".mp3", ".jpg", ".png", ".webm"]

    # Monitoring
    ENABLE_METRICS: bool = True
    SENTRY_DSN: Optional[str] = None
    LOG_LEVEL: str = "INFO"

    # Email (for notifications)
    SMTP_HOST: Optional[str] = None
    SMTP_PORT: Optional[int] = 587
    SMTP_USER: Optional[str] = None
    SMTP_PASSWORD: Optional[str] = None
    EMAILS_FROM_EMAIL: Optional[EmailStr] = None
    EMAILS_FROM_NAME: Optional[str] = "YTEmpire"

    # Admin
    FIRST_SUPERUSER: EmailStr = "admin@ytempire.com"
    FIRST_SUPERUSER_PASSWORD: str = "changeme123!"

    # N8N Workflow
    N8N_URL: Optional[str] = None
    N8N_PASSWORD: Optional[str] = None
    N8N_WEBHOOK_URL: Optional[str] = None
    N8N_API_KEY: Optional[str] = None

    # Payment (Stripe)
    STRIPE_SECRET_KEY: Optional[str] = None
    STRIPE_PUBLISHABLE_KEY: Optional[str] = None
    STRIPE_WEBHOOK_SECRET: Optional[str] = None

    # ML Configuration
    ML_MODELS_PATH: str = "models"
    AUTOML_RETRAIN_DAYS: int = 7
    PERSONALIZATION_UPDATE_DAYS: int = 3
    ML_ENABLED: bool = True

    @property
    def database_pool(self) -> Dict[str, Any]:
        """Expose database pool configuration as a dictionary"""
        return {
            "pool_size": self.DATABASE_POOL_SIZE,
            "max_overflow": self.DATABASE_MAX_OVERFLOW,
            "pool_timeout": self.DATABASE_POOL_TIMEOUT,
            "pool_recycle": self.DATABASE_POOL_RECYCLE,
            "echo": self.DATABASE_ECHO,
        }

    class Config:
        case_sensitive = True
        env_file = ".env"
        env_file_encoding = "utf-8"
        extra = "ignore"  # Ignore extra fields in .env file


settings = Settings()
