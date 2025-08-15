"""
Multi-Provider AI Service API Endpoints
Claude API, Stable Diffusion, Azure TTS, and other AI providers
"""
from typing import List, Optional, Dict, Any, Union
from fastapi import APIRouter, Depends, HTTPException, status, Query, BackgroundTasks
from sqlalchemy.ext.asyncio import AsyncSession
from pydantic import BaseModel, Field
from datetime import datetime
from enum import Enum

from app.db.session import get_db
from app.api.v1.endpoints.auth import get_current_verified_user
from app.models.user import User
from app.services.multi_provider_ai import multi_ai_service, AIProvider, ServiceType
from app.services.cost_tracking import cost_tracker

import logging

logger = logging.getLogger(__name__)

router = APIRouter()


class TextGenerationRequest(BaseModel):
    """Request for text generation"""
    prompt: str = Field(..., min_length=1, max_length=10000)
    provider: Optional[AIProvider] = None
    max_tokens: int = Field(1000, ge=1, le=4000)
    temperature: float = Field(0.7, ge=0.0, le=2.0)
    system_prompt: Optional[str] = None


class SpeechSynthesisRequest(BaseModel):
    """Request for speech synthesis"""
    text: str = Field(..., min_length=1, max_length=5000)
    provider: Optional[AIProvider] = None
    voice: str = Field("default", description="Voice identifier")
    language: str = Field("en-US", description="Language code")
    speed: float = Field(1.0, ge=0.5, le=2.0)
    pitch: float = Field(0.0, ge=-20.0, le=20.0)


class ImageGenerationRequest(BaseModel):
    """Request for image generation"""
    prompt: str = Field(..., min_length=1, max_length=1000)
    provider: Optional[AIProvider] = None
    size: str = Field("1024x1024", regex="^(256x256|512x512|1024x1024|1792x1024|1024x1792)$")
    style: str = Field("realistic", description="Image style: realistic, artistic, cartoon")
    num_images: int = Field(1, ge=1, le=4)


class TranslationRequest(BaseModel):
    """Request for text translation"""
    text: str = Field(..., min_length=1, max_length=5000)
    target_language: str = Field(..., description="Target language code (e.g., 'es', 'fr', 'de')")
    source_language: str = Field("auto", description="Source language code or 'auto' for detection")
    provider: Optional[AIProvider] = None


class SummarizationRequest(BaseModel):
    """Request for text summarization"""
    text: str = Field(..., min_length=100, max_length=50000)
    max_length: int = Field(150, ge=50, le=1000)
    style: str = Field("bullet_points", description="Summary style: bullet_points, paragraph, keywords")
    provider: Optional[AIProvider] = None


class SentimentAnalysisRequest(BaseModel):
    """Request for sentiment analysis"""
    text: str = Field(..., min_length=1, max_length=5000)
    provider: Optional[AIProvider] = None
    detailed: bool = Field(False, description="Include detailed emotion analysis")


class BatchProcessingRequest(BaseModel):
    """Request for batch AI processing"""
    requests: List[Dict[str, Any]] = Field(..., min_items=1, max_items=10)
    service_type: ServiceType
    provider: Optional[AIProvider] = None
    parallel_processing: bool = Field(True, description="Process requests in parallel")


@router.post("/text/generate")
async def generate_text(
    request: TextGenerationRequest,
    background_tasks: BackgroundTasks,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_verified_user)
) -> Dict[str, Any]:
    """
    Generate text using AI providers (OpenAI, Claude, etc.)
    """
    try:
        # Combine prompts if system prompt provided
        full_prompt = request.prompt
        if request.system_prompt:
            full_prompt = f"System: {request.system_prompt}\n\nUser: {request.prompt}"
        
        response = await multi_ai_service.generate_text(
            prompt=full_prompt,
            provider=request.provider,
            max_tokens=request.max_tokens,
            temperature=request.temperature,
            user_id=str(current_user.id)
        )
        
        if not response.success:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=response.error
            )
        
        return {
            "success": True,
            "generated_text": response.result,
            "provider_used": response.provider.value,
            "cost": response.cost,
            "latency_ms": response.latency_ms,
            "metadata": {
                "input_length": len(request.prompt),
                "output_length": len(response.result),
                "model_parameters": {
                    "max_tokens": request.max_tokens,
                    "temperature": request.temperature
                }
            }
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Text generation failed: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Text generation failed: {str(e)}"
        )


@router.post("/speech/synthesize")
async def synthesize_speech(
    request: SpeechSynthesisRequest,
    background_tasks: BackgroundTasks,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_verified_user)
) -> Dict[str, Any]:
    """
    Convert text to speech using AI providers (Azure TTS, Google, etc.)
    """
    try:
        response = await multi_ai_service.synthesize_speech(
            text=request.text,
            voice=request.voice,
            provider=request.provider,
            language=request.language,
            user_id=str(current_user.id)
        )
        
        if not response.success:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=response.error
            )
        
        # In production, you would save the audio to storage and return URL
        audio_url = f"https://storage.ytempire.com/audio/{current_user.id}/{datetime.utcnow().timestamp()}.mp3"
        
        return {
            "success": True,
            "audio_url": audio_url,
            "provider_used": response.provider.value,
            "cost": response.cost,
            "latency_ms": response.latency_ms,
            "audio_details": {
                "duration_seconds": response.result.get("audio_duration", 0),
                "voice_used": request.voice,
                "language": request.language,
                "text_length": len(request.text)
            }
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Speech synthesis failed: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Speech synthesis failed: {str(e)}"
        )


@router.post("/images/generate")
async def generate_image(
    request: ImageGenerationRequest,
    background_tasks: BackgroundTasks,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_verified_user)
) -> Dict[str, Any]:
    """
    Generate images using AI providers (DALL-E, Stable Diffusion, etc.)
    """
    try:
        response = await multi_ai_service.generate_image(
            prompt=request.prompt,
            size=request.size,
            provider=request.provider,
            style=request.style,
            user_id=str(current_user.id)
        )
        
        if not response.success:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=response.error
            )
        
        # Process response based on provider
        if isinstance(response.result, dict):
            image_url = response.result.get("image_url")
            metadata = {
                "revised_prompt": response.result.get("revised_prompt"),
                "seed": response.result.get("seed"),
                "steps": response.result.get("steps")
            }
        else:
            image_url = str(response.result)
            metadata = {}
        
        return {
            "success": True,
            "image_url": image_url,
            "provider_used": response.provider.value,
            "cost": response.cost,
            "latency_ms": response.latency_ms,
            "image_details": {
                "prompt": request.prompt,
                "size": request.size,
                "style": request.style,
                **metadata
            }
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Image generation failed: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Image generation failed: {str(e)}"
        )


@router.post("/text/translate")
async def translate_text(
    request: TranslationRequest,
    background_tasks: BackgroundTasks,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_verified_user)
) -> Dict[str, Any]:
    """
    Translate text between languages using AI providers
    """
    try:
        response = await multi_ai_service.translate_text(
            text=request.text,
            target_language=request.target_language,
            source_language=request.source_language,
            provider=request.provider,
            user_id=str(current_user.id)
        )
        
        if not response.success:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=response.error
            )
        
        # Extract translation result
        if isinstance(response.result, dict):
            translated_text = response.result.get("translated_text", response.result)
            detected_language = response.result.get("detected_language")
            confidence = response.result.get("confidence")
        else:
            translated_text = str(response.result)
            detected_language = None
            confidence = None
        
        return {
            "success": True,
            "translated_text": translated_text,
            "provider_used": response.provider.value,
            "cost": response.cost,
            "latency_ms": response.latency_ms,
            "translation_details": {
                "source_language": detected_language or request.source_language,
                "target_language": request.target_language,
                "confidence": confidence,
                "original_length": len(request.text),
                "translated_length": len(translated_text)
            }
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Translation failed: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Translation failed: {str(e)}"
        )


@router.post("/text/summarize")
async def summarize_text(
    request: SummarizationRequest,
    background_tasks: BackgroundTasks,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_verified_user)
) -> Dict[str, Any]:
    """
    Summarize long text using AI providers
    """
    try:
        # Enhance prompt based on style
        enhanced_prompt = request.text
        if request.style == "bullet_points":
            enhanced_prompt = f"Summarize the following text in bullet points (max {request.max_length} words):\\n\\n{request.text}"
        elif request.style == "paragraph":
            enhanced_prompt = f"Provide a concise paragraph summary (max {request.max_length} words):\\n\\n{request.text}"
        elif request.style == "keywords":
            enhanced_prompt = f"Extract key points and keywords from the following text:\\n\\n{request.text}"
        
        response = await multi_ai_service.summarize_text(
            text=enhanced_prompt,
            max_length=request.max_length,
            provider=request.provider,
            user_id=str(current_user.id)
        )
        
        if not response.success:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=response.error
            )
        
        return {
            "success": True,
            "summary": response.result,
            "provider_used": response.provider.value,
            "cost": response.cost,
            "latency_ms": response.latency_ms,
            "summary_details": {
                "original_length": len(request.text),
                "summary_length": len(response.result),
                "compression_ratio": len(response.result) / len(request.text),
                "style": request.style,
                "max_length": request.max_length
            }
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Summarization failed: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Summarization failed: {str(e)}"
        )


@router.post("/text/sentiment")
async def analyze_sentiment(
    request: SentimentAnalysisRequest,
    background_tasks: BackgroundTasks,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_verified_user)
) -> Dict[str, Any]:
    """
    Analyze sentiment of text using AI providers
    """
    try:
        response = await multi_ai_service.analyze_sentiment(
            text=request.text,
            provider=request.provider,
            user_id=str(current_user.id)
        )
        
        if not response.success:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=response.error
            )
        
        # Extract sentiment results
        if isinstance(response.result, dict):
            sentiment = response.result.get("sentiment", "NEUTRAL")
            confidence = response.result.get("confidence", 0.0)
            magnitude = response.result.get("magnitude", 0.0)
        else:
            sentiment = str(response.result)
            confidence = 0.0
            magnitude = 0.0
        
        return {
            "success": True,
            "sentiment": sentiment,
            "confidence": confidence,
            "magnitude": magnitude,
            "provider_used": response.provider.value,
            "cost": response.cost,
            "latency_ms": response.latency_ms,
            "analysis_details": {
                "text_length": len(request.text),
                "detailed_analysis": request.detailed,
                "sentiment_category": _categorize_sentiment(sentiment, confidence)
            }
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Sentiment analysis failed: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Sentiment analysis failed: {str(e)}"
        )


@router.post("/batch/process")
async def process_batch_requests(
    request: BatchProcessingRequest,
    background_tasks: BackgroundTasks,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_verified_user)
) -> Dict[str, Any]:
    """
    Process multiple AI requests in batch for improved efficiency
    """
    try:
        batch_id = f"batch_{datetime.utcnow().timestamp()}"
        results = []
        total_cost = 0
        
        # Process requests (simplified implementation)
        for i, req_data in enumerate(request.requests):
            try:
                # Determine the type of request and route appropriately
                if request.service_type == ServiceType.TEXT_GENERATION:
                    response = await multi_ai_service.generate_text(
                        prompt=req_data.get("prompt", ""),
                        provider=request.provider,
                        user_id=str(current_user.id)
                    )
                elif request.service_type == ServiceType.TEXT_TO_SPEECH:
                    response = await multi_ai_service.synthesize_speech(
                        text=req_data.get("text", ""),
                        provider=request.provider,
                        user_id=str(current_user.id)
                    )
                elif request.service_type == ServiceType.IMAGE_GENERATION:
                    response = await multi_ai_service.generate_image(
                        prompt=req_data.get("prompt", ""),
                        provider=request.provider,
                        user_id=str(current_user.id)
                    )
                else:
                    response = None
                
                if response and response.success:
                    results.append({
                        "request_index": i,
                        "success": True,
                        "result": response.result,
                        "provider": response.provider.value,
                        "cost": response.cost,
                        "latency_ms": response.latency_ms
                    })
                    total_cost += response.cost
                else:
                    results.append({
                        "request_index": i,
                        "success": False,
                        "error": response.error if response else "Unknown error",
                        "cost": 0,
                        "latency_ms": 0
                    })
                    
            except Exception as e:
                results.append({
                    "request_index": i,
                    "success": False,
                    "error": str(e),
                    "cost": 0,
                    "latency_ms": 0
                })
        
        # Track total batch cost
        background_tasks.add_task(
            cost_tracker.track_cost,
            user_id=str(current_user.id),
            service="ai_batch_processing",
            operation=f"batch_{request.service_type.value}",
            cost=total_cost,
            metadata={
                "batch_id": batch_id,
                "request_count": len(request.requests),
                "success_count": len([r for r in results if r["success"]])
            }
        )
        
        return {
            "success": True,
            "batch_id": batch_id,
            "total_requests": len(request.requests),
            "successful_requests": len([r for r in results if r["success"]]),
            "failed_requests": len([r for r in results if not r["success"]]),
            "total_cost": total_cost,
            "results": results,
            "processing_details": {
                "service_type": request.service_type.value,
                "provider_used": request.provider.value if request.provider else "auto",
                "parallel_processing": request.parallel_processing,
                "average_latency_ms": sum(r["latency_ms"] for r in results) / len(results) if results else 0
            }
        }
        
    except Exception as e:
        logger.error(f"Batch processing failed: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Batch processing failed: {str(e)}"
        )


@router.get("/providers/status")
async def get_providers_status(
    current_user: User = Depends(get_current_verified_user)
) -> Dict[str, Any]:
    """
    Get status and availability of all AI providers
    """
    try:
        status = await multi_ai_service.get_provider_status()
        
        return {
            "success": True,
            "provider_status": status,
            "summary": {
                "total_providers": status["total_providers"],
                "available_providers": len([p for p in status["providers"].values() if p["available"]]),
                "service_coverage": {
                    service: len(info["available_providers"])
                    for service, info in status["services"].items()
                }
            },
            "checked_at": datetime.utcnow().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Provider status check failed: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to get provider status: {str(e)}"
        )


@router.post("/providers/{provider_name}/test")
async def test_provider_connectivity(
    provider_name: str,
    current_user: User = Depends(get_current_verified_user)
) -> Dict[str, Any]:
    """
    Test connectivity to a specific AI provider
    """
    try:
        # Validate provider name
        try:
            provider = AIProvider(provider_name)
        except ValueError:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Invalid provider: {provider_name}"
            )
        
        test_result = await multi_ai_service.test_provider_connectivity(provider)
        
        return {
            "success": True,
            "connectivity_test": test_result,
            "tested_at": datetime.utcnow().isoformat()
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Provider connectivity test failed: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Connectivity test failed: {str(e)}"
        )


@router.get("/providers/costs")
async def get_provider_costs(
    service_type: Optional[ServiceType] = Query(None, description="Filter by service type"),
    current_user: User = Depends(get_current_verified_user)
) -> Dict[str, Any]:
    """
    Get cost information for different providers and services
    """
    try:
        # Cost comparison data (in production, this would be dynamic)
        cost_data = {
            "cost_comparison": {
                AIProvider.OPENAI.value: {
                    ServiceType.TEXT_GENERATION.value: {
                        "cost_per_1k_tokens": 0.003,
                        "quality_score": 90,
                        "avg_latency_ms": 800
                    },
                    ServiceType.IMAGE_GENERATION.value: {
                        "cost_per_image": 0.02,
                        "quality_score": 95,
                        "avg_latency_ms": 10000
                    }
                },
                AIProvider.ANTHROPIC.value: {
                    ServiceType.TEXT_GENERATION.value: {
                        "cost_per_1k_tokens": 0.015,
                        "quality_score": 92,
                        "avg_latency_ms": 1000
                    },
                    ServiceType.SUMMARIZATION.value: {
                        "cost_per_1k_tokens": 0.015,
                        "quality_score": 93,
                        "avg_latency_ms": 1200
                    }
                },
                AIProvider.AZURE.value: {
                    ServiceType.TEXT_TO_SPEECH.value: {
                        "cost_per_1k_chars": 0.004,
                        "quality_score": 88,
                        "avg_latency_ms": 2000
                    }
                },
                AIProvider.GOOGLE.value: {
                    ServiceType.TEXT_TO_SPEECH.value: {
                        "cost_per_1k_chars": 0.002,
                        "quality_score": 85,
                        "avg_latency_ms": 1500
                    },
                    ServiceType.TRANSLATION.value: {
                        "cost_per_1k_chars": 0.008,
                        "quality_score": 90,
                        "avg_latency_ms": 500
                    }
                }
            },
            "recommendations": {
                "most_cost_effective": {
                    ServiceType.TEXT_GENERATION.value: AIProvider.OPENAI.value,
                    ServiceType.TEXT_TO_SPEECH.value: AIProvider.GOOGLE.value,
                    ServiceType.IMAGE_GENERATION.value: AIProvider.OPENAI.value,
                    ServiceType.TRANSLATION.value: AIProvider.GOOGLE.value
                },
                "highest_quality": {
                    ServiceType.TEXT_GENERATION.value: AIProvider.ANTHROPIC.value,
                    ServiceType.TEXT_TO_SPEECH.value: AIProvider.AZURE.value,
                    ServiceType.IMAGE_GENERATION.value: AIProvider.OPENAI.value,
                    ServiceType.TRANSLATION.value: AIProvider.GOOGLE.value
                },
                "fastest_response": {
                    ServiceType.TEXT_GENERATION.value: AIProvider.OPENAI.value,
                    ServiceType.TEXT_TO_SPEECH.value: AIProvider.GOOGLE.value,
                    ServiceType.TRANSLATION.value: AIProvider.GOOGLE.value
                }
            }
        }
        
        # Filter by service type if requested
        if service_type:
            filtered_costs = {}
            for provider, services in cost_data["cost_comparison"].items():
                if service_type.value in services:
                    filtered_costs[provider] = {service_type.value: services[service_type.value]}
            cost_data["cost_comparison"] = filtered_costs
        
        return {
            "success": True,
            "pricing_data": cost_data,
            "notes": [
                "Costs are estimates and may vary based on actual usage",
                "Quality scores are based on internal benchmarks",
                "Latency varies by region and current load"
            ],
            "updated_at": datetime.utcnow().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Provider costs retrieval failed: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to get provider costs: {str(e)}"
        )


# Helper functions

def _categorize_sentiment(sentiment: str, confidence: float) -> str:
    """Categorize sentiment with confidence levels"""
    if confidence < 0.6:
        return "uncertain"
    elif sentiment.upper() == "POSITIVE":
        return "positive" if confidence > 0.8 else "somewhat_positive"
    elif sentiment.upper() == "NEGATIVE":
        return "negative" if confidence > 0.8 else "somewhat_negative"
    else:
        return "neutral"