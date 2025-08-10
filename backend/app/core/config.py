"""
Core Configuration
Owner: API Developer
"""

from typing import List, Optional
from pydantic_settings import BaseSettings
from pydantic import AnyHttpUrl, validator
import os
import asyncio
import logging

logger = logging.getLogger(__name__)


class Settings(BaseSettings):
    # Project Info
    PROJECT_NAME: str = "YTEmpire"
    VERSION: str = "0.1.0"
    API_V1_STR: str = "/api/v1"
    DEBUG: bool = True
    
    # HashiCorp Vault
    VAULT_URL: str = os.getenv("VAULT_URL", "http://localhost:8200")
    VAULT_TOKEN: Optional[str] = os.getenv("VAULT_TOKEN", None)
    USE_VAULT: bool = os.getenv("USE_VAULT", "true").lower() == "true"
    
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
    
    # N8N Configuration (Integration Specialist)
    N8N_BASE_URL: str = os.getenv("N8N_BASE_URL", "http://n8n:5678")
    N8N_WEBHOOK_URL: str = os.getenv("N8N_WEBHOOK_URL", "http://localhost:5678/webhook")
    N8N_API_KEY: Optional[str] = os.getenv("N8N_API_KEY", None)
    N8N_WEBHOOK_SECRET: str = os.getenv("N8N_WEBHOOK_SECRET", "ytempire-n8n-secret")
    BACKEND_URL: str = os.getenv("BACKEND_URL", "http://backend:8000")
    
    # Monitoring (Platform Ops Lead)
    PROMETHEUS_ENABLED: bool = True
    GRAFANA_URL: str = "http://localhost:3001"
    
    # Vector Database Configuration
    QDRANT_HOST: str = os.getenv("QDRANT_HOST", "localhost")
    QDRANT_PORT: int = int(os.getenv("QDRANT_PORT", "6333"))
    QDRANT_GRPC_PORT: int = int(os.getenv("QDRANT_GRPC_PORT", "6334"))
    QDRANT_API_KEY: Optional[str] = os.getenv("QDRANT_API_KEY", None)
    QDRANT_URL: str = f"http://{QDRANT_HOST}:{QDRANT_PORT}"
    
    # Embedding Configuration
    EMBEDDING_MODEL: str = "all-MiniLM-L6-v2"
    EMBEDDING_DIMENSION: int = 384
    SIMILARITY_THRESHOLD: float = 0.7
    
    def load_from_vault(self):
        """Load configuration from Vault if enabled."""
        if not self.USE_VAULT:
            return
        
        try:
            from app.services.vault_service import get_app_config, get_api_keys
            
            # This is a synchronous context, so we need to handle async calls carefully
            # In production, this should be called from an async context during app startup
            loop = None
            try:
                loop = asyncio.get_event_loop()
                if loop.is_running():
                    # If loop is already running, we can't use run_until_complete
                    logger.warning("Cannot load Vault config in running event loop. Load manually during startup.")
                    return
            except RuntimeError:
                # No event loop, create one
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)
            
            app_config = loop.run_until_complete(get_app_config())
            api_keys = loop.run_until_complete(get_api_keys())
            
            # Update configuration with Vault values
            if app_config:
                if 'secret_key' in app_config:
                    self.SECRET_KEY = app_config['secret_key']
                if 'database_url' in app_config:
                    self.DATABASE_URL = app_config['database_url']
                if 'redis_url' in app_config:
                    self.REDIS_URL = app_config['redis_url']
            
            if api_keys:
                if 'openai_api_key' in api_keys:
                    self.OPENAI_API_KEY = api_keys['openai_api_key']
                if 'elevenlabs_api_key' in api_keys:
                    self.ELEVENLABS_API_KEY = api_keys['elevenlabs_api_key']
                if 'youtube_client_id' in api_keys:
                    self.YOUTUBE_CLIENT_ID = api_keys['youtube_client_id']
                if 'youtube_client_secret' in api_keys:
                    self.YOUTUBE_CLIENT_SECRET = api_keys['youtube_client_secret']
            
            logger.info("Configuration loaded from Vault successfully")
            
        except Exception as e:
            logger.warning(f"Failed to load configuration from Vault: {str(e)}")
            logger.warning("Falling back to environment variables")

    class Config:
        case_sensitive = True
        env_file = ".env"


# Initialize settings
settings = Settings()

# Load from Vault if enabled (will be called properly during app startup)
if settings.USE_VAULT:
    # This will be handled properly in main.py during app startup
    pass