"""
AI Services Configuration
Owner: VP of AI
Budget Target: <$3 per video
"""

import os
from typing import Dict, Any
from dataclasses import dataclass

@dataclass
class AIServiceConfig:
    """Configuration for AI services with cost optimization"""
    
    # OpenAI Configuration ($5000 initial credit)
    OPENAI_API_KEY: str = os.getenv("OPENAI_API_KEY", "")
    OPENAI_ORG_ID: str = os.getenv("OPENAI_ORG_ID", "")
    
    # Model Selection (Cost Optimization)
    SCRIPT_MODEL: str = "gpt-3.5-turbo-1106"  # $0.001/$0.002 per 1K tokens
    SCRIPT_FALLBACK_MODEL: str = "gpt-3.5-turbo"  # Fallback option
    HIGH_QUALITY_MODEL: str = "gpt-4-1106-preview"  # $0.01/$0.03 - only for premium
    
    # ElevenLabs Configuration
    ELEVENLABS_API_KEY: str = os.getenv("ELEVENLABS_API_KEY", "")
    ELEVENLABS_VOICE_ID: str = "21m00Tcm4TlvDq8ikWAM"  # Rachel voice
    ELEVENLABS_MODEL: str = "eleven_monolingual_v1"  # Cheaper option
    
    # Google Cloud TTS (Backup)
    GOOGLE_TTS_KEY: str = os.getenv("GOOGLE_CLOUD_TTS_KEY", "")
    GOOGLE_TTS_VOICE: str = "en-US-Neural2-F"
    
    # Cost Limits per Component (Total must be <$3)
    MAX_SCRIPT_COST: float = 0.50  # Script generation
    MAX_VOICE_COST: float = 1.00   # Voice synthesis
    MAX_IMAGE_COST: float = 0.50   # Image generation/processing
    MAX_VIDEO_COST: float = 0.50   # Video processing
    MAX_MUSIC_COST: float = 0.30   # Background music
    MAX_OTHER_COST: float = 0.20   # Other services
    
    # Token Limits
    MAX_SCRIPT_TOKENS: int = 2000  # ~1500 words
    MAX_TITLE_TOKENS: int = 100
    MAX_DESCRIPTION_TOKENS: int = 300
    MAX_TAGS_TOKENS: int = 150
    
    # Quality Settings
    VOICE_STABILITY: float = 0.5
    VOICE_SIMILARITY: float = 0.75
    VOICE_STYLE: float = 0.0
    VOICE_SPEAKER_BOOST: bool = True
    
    # Retry Configuration
    MAX_RETRIES: int = 3
    RETRY_DELAY: int = 2  # seconds
    
    # Cache Settings
    ENABLE_CACHE: bool = True
    CACHE_TTL: int = 86400  # 24 hours


@dataclass
class CostTracker:
    """Track costs per video generation"""
    
    def __init__(self):
        self.script_cost: float = 0.0
        self.voice_cost: float = 0.0
        self.image_cost: float = 0.0
        self.video_cost: float = 0.0
        self.music_cost: float = 0.0
        self.other_cost: float = 0.0
    
    @property
    def total_cost(self) -> float:
        return (
            self.script_cost +
            self.voice_cost +
            self.image_cost +
            self.video_cost +
            self.music_cost +
            self.other_cost
        )
    
    def is_within_budget(self, max_cost: float = 3.0) -> bool:
        return self.total_cost <= max_cost
    
    def get_breakdown(self) -> Dict[str, float]:
        return {
            "script": self.script_cost,
            "voice": self.voice_cost,
            "image": self.image_cost,
            "video": self.video_cost,
            "music": self.music_cost,
            "other": self.other_cost,
            "total": self.total_cost
        }


# Optimization Strategies (VP of AI requirement)
class CostOptimizer:
    """Strategies to keep costs under $3/video"""
    
    @staticmethod
    def select_model(quality_tier: str = "standard") -> str:
        """Select appropriate model based on quality tier"""
        models = {
            "economy": "gpt-3.5-turbo",  # Cheapest
            "standard": "gpt-3.5-turbo-1106",  # Good balance
            "premium": "gpt-4-1106-preview"  # High quality
        }
        return models.get(quality_tier, models["standard"])
    
    @staticmethod
    def optimize_tokens(text: str, max_tokens: int) -> str:
        """Optimize token usage"""
        # Simple truncation for now
        words = text.split()
        if len(words) > max_tokens * 0.75:  # Rough estimate
            return " ".join(words[:int(max_tokens * 0.75)])
        return text
    
    @staticmethod
    def should_use_cache(content_type: str, niche: str) -> bool:
        """Determine if content should be cached"""
        # Cache evergreen content
        cacheable_niches = ["educational", "how-to", "facts", "history"]
        return niche.lower() in cacheable_niches
    
    @staticmethod
    def select_voice_service(budget_remaining: float) -> str:
        """Choose voice service based on remaining budget"""
        if budget_remaining > 1.5:
            return "elevenlabs"  # Better quality
        elif budget_remaining > 0.5:
            return "google_tts"  # Cheaper
        else:
            return "local_tts"  # Free but lower quality


# Export configuration
ai_config = AIServiceConfig()
cost_optimizer = CostOptimizer()