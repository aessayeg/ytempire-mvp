"""
Multi-Provider AI Service Integration
Supports Claude API, Stable Diffusion, Azure TTS, and other AI providers
"""
import os
import json
import logging
import asyncio
from typing import Dict, Any, List, Optional, Union
from datetime import datetime
from enum import Enum
from dataclasses import dataclass
import aiohttp
import openai
from anthropic import Anthropic

# Azure imports
try:
    import azure.cognitiveservices.speech as speechsdk
    AZURE_AVAILABLE = True
except ImportError:
    AZURE_AVAILABLE = False

# Stability AI imports
try:
    import stability_sdk.interfaces.gooseai.generation.generation_pb2 as generation
    from stability_sdk import client
    STABILITY_AVAILABLE = True
except ImportError:
    STABILITY_AVAILABLE = False

from app.services.cost_tracking import cost_tracker

logger = logging.getLogger(__name__)


class AIProvider(str, Enum):
    """Available AI providers"""
    OPENAI = "openai"
    ANTHROPIC = "anthropic"
    AZURE = "azure"
    STABILITY = "stability"
    GOOGLE = "google"
    HUGGINGFACE = "huggingface"


class ServiceType(str, Enum):
    """AI service types"""
    TEXT_GENERATION = "text_generation"
    TEXT_TO_SPEECH = "text_to_speech"
    IMAGE_GENERATION = "image_generation"
    TRANSLATION = "translation"
    SUMMARIZATION = "summarization"
    SENTIMENT_ANALYSIS = "sentiment_analysis"


@dataclass
class AIRequest:
    """AI service request"""
    service_type: ServiceType
    provider: AIProvider
    input_text: str
    parameters: Dict[str, Any]
    user_id: str
    metadata: Dict[str, Any] = None


@dataclass
class AIResponse:
    """AI service response"""
    success: bool
    result: Any
    provider: AIProvider
    service_type: ServiceType
    cost: float
    latency_ms: int
    metadata: Dict[str, Any] = None
    error: str = None


class MultiProviderAIService:
    """
    Multi-provider AI service with intelligent routing and fallback
    """
    
    def __init__(self):
        """Initialize multi-provider AI service"""
        self.providers = {}
        self.fallback_chains = {}
        self.cost_thresholds = {}
        self.quality_scores = {}
        
        # Initialize providers
        self._initialize_providers()
        self._setup_fallback_chains()
        self._setup_cost_thresholds()
        
    def _initialize_providers(self):
        """Initialize all available AI providers"""
        
        # OpenAI
        if os.getenv("OPENAI_API_KEY"):
            self.providers[AIProvider.OPENAI] = {
                "client": openai.AsyncOpenAI(api_key=os.getenv("OPENAI_API_KEY")),
                "available": True,
                "services": [ServiceType.TEXT_GENERATION, ServiceType.IMAGE_GENERATION]
            }
            logger.info("OpenAI provider initialized")
        
        # Anthropic (Claude)
        if os.getenv("ANTHROPIC_API_KEY"):
            self.providers[AIProvider.ANTHROPIC] = {
                "client": Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY")),
                "available": True,
                "services": [ServiceType.TEXT_GENERATION, ServiceType.SUMMARIZATION]
            }
            logger.info("Anthropic (Claude) provider initialized")
        
        # Azure Cognitive Services
        if os.getenv("AZURE_SPEECH_KEY") and AZURE_AVAILABLE:
            self.providers[AIProvider.AZURE] = {
                "speech_key": os.getenv("AZURE_SPEECH_KEY"),
                "speech_region": os.getenv("AZURE_SPEECH_REGION", "eastus"),
                "available": True,
                "services": [ServiceType.TEXT_TO_SPEECH, ServiceType.TRANSLATION]
            }
            logger.info("Azure Cognitive Services provider initialized")
        
        # Stability AI (Stable Diffusion)
        if os.getenv("STABILITY_API_KEY") and STABILITY_AVAILABLE:
            self.providers[AIProvider.STABILITY] = {
                "api_key": os.getenv("STABILITY_API_KEY"),
                "available": True,
                "services": [ServiceType.IMAGE_GENERATION]
            }
            logger.info("Stability AI provider initialized")
        
        # Google Cloud AI
        if os.getenv("GOOGLE_APPLICATION_CREDENTIALS"):
            self.providers[AIProvider.GOOGLE] = {
                "available": True,
                "services": [ServiceType.TEXT_TO_SPEECH, ServiceType.TRANSLATION, ServiceType.SENTIMENT_ANALYSIS]
            }
            logger.info("Google Cloud AI provider initialized")
    
    def _setup_fallback_chains(self):
        """Setup fallback chains for each service type"""
        self.fallback_chains = {
            ServiceType.TEXT_GENERATION: [
                AIProvider.OPENAI,
                AIProvider.ANTHROPIC,
                AIProvider.HUGGINGFACE
            ],
            ServiceType.TEXT_TO_SPEECH: [
                AIProvider.AZURE,
                AIProvider.GOOGLE,
                AIProvider.OPENAI
            ],
            ServiceType.IMAGE_GENERATION: [
                AIProvider.STABILITY,
                AIProvider.OPENAI,
                AIProvider.HUGGINGFACE
            ],
            ServiceType.TRANSLATION: [
                AIProvider.GOOGLE,
                AIProvider.AZURE,
                AIProvider.OPENAI
            ],
            ServiceType.SUMMARIZATION: [
                AIProvider.ANTHROPIC,
                AIProvider.OPENAI,
                AIProvider.GOOGLE
            ],
            ServiceType.SENTIMENT_ANALYSIS: [
                AIProvider.GOOGLE,
                AIProvider.AZURE,
                AIProvider.HUGGINGFACE
            ]
        }
    
    def _setup_cost_thresholds(self):
        """Setup cost thresholds for each provider"""
        self.cost_thresholds = {
            AIProvider.OPENAI: {
                ServiceType.TEXT_GENERATION: 0.03,  # per 1k tokens
                ServiceType.IMAGE_GENERATION: 0.02   # per image
            },
            AIProvider.ANTHROPIC: {
                ServiceType.TEXT_GENERATION: 0.015,  # per 1k tokens
                ServiceType.SUMMARIZATION: 0.015
            },
            AIProvider.AZURE: {
                ServiceType.TEXT_TO_SPEECH: 0.004,   # per 1k chars
                ServiceType.TRANSLATION: 0.01       # per 1k chars
            },
            AIProvider.STABILITY: {
                ServiceType.IMAGE_GENERATION: 0.01   # per image
            },
            AIProvider.GOOGLE: {
                ServiceType.TEXT_TO_SPEECH: 0.002,   # per 1k chars
                ServiceType.TRANSLATION: 0.008,      # per 1k chars
                ServiceType.SENTIMENT_ANALYSIS: 0.001
            }
        }
    
    async def generate_text(
        self,
        prompt: str,
        provider: Optional[AIProvider] = None,
        max_tokens: int = 1000,
        temperature: float = 0.7,
        user_id: str = None
    ) -> AIResponse:
        """Generate text using specified or optimal provider"""
        
        request = AIRequest(
            service_type=ServiceType.TEXT_GENERATION,
            provider=provider or self._get_optimal_provider(ServiceType.TEXT_GENERATION),
            input_text=prompt,
            parameters={
                "max_tokens": max_tokens,
                "temperature": temperature
            },
            user_id=user_id or "unknown"
        )
        
        return await self._execute_request(request)
    
    async def synthesize_speech(
        self,
        text: str,
        voice: str = "default",
        provider: Optional[AIProvider] = None,
        language: str = "en-US",
        user_id: str = None
    ) -> AIResponse:
        """Synthesize speech from text"""
        
        request = AIRequest(
            service_type=ServiceType.TEXT_TO_SPEECH,
            provider=provider or self._get_optimal_provider(ServiceType.TEXT_TO_SPEECH),
            input_text=text,
            parameters={
                "voice": voice,
                "language": language
            },
            user_id=user_id or "unknown"
        )
        
        return await self._execute_request(request)
    
    async def generate_image(
        self,
        prompt: str,
        size: str = "1024x1024",
        provider: Optional[AIProvider] = None,
        style: str = "realistic",
        user_id: str = None
    ) -> AIResponse:
        """Generate image from text prompt"""
        
        request = AIRequest(
            service_type=ServiceType.IMAGE_GENERATION,
            provider=provider or self._get_optimal_provider(ServiceType.IMAGE_GENERATION),
            input_text=prompt,
            parameters={
                "size": size,
                "style": style
            },
            user_id=user_id or "unknown"
        )
        
        return await self._execute_request(request)