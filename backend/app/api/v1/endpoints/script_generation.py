"""
Script Generation API Endpoints
Handles AI-powered video script generation
"""
from typing import List, Optional, Dict, Any
from fastapi import APIRouter, Depends, HTTPException, status, BackgroundTasks
from sqlalchemy.ext.asyncio import AsyncSession
from pydantic import BaseModel, Field
from datetime import datetime
import os
import sys

# Add ml-pipeline to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..', '..', '..', '..', 'ml-pipeline', 'src'))

from app.db.session import get_db
from app.core.security import get_current_user
from app.models.user import User
from app.services.cost_tracking import CostTrackingService
from script_generation import ScriptGenerator, ScriptRequest, ScriptStyle, ScriptTone
import logging

logger = logging.getLogger(__name__)

router = APIRouter()

# Initialize script generator
openai_key = os.getenv("OPENAI_API_KEY")
if openai_key:
    script_generator = ScriptGenerator(api_key=openai_key)
else:
    script_generator = None
    logger.warning("OpenAI API key not configured - script generation disabled")


class ScriptGenerationRequest(BaseModel):
    """Request model for script generation"""
    topic: str = Field(..., description="Video topic or title")
    style: str = Field(default="informative", description="Script style")
    tone: str = Field(default="professional", description="Script tone")
    duration_minutes: int = Field(default=5, ge=1, le=30, description="Target video duration in minutes")
    target_audience: str = Field(default="general", description="Target audience description")
    keywords: List[str] = Field(default=[], description="Keywords to include for SEO")
    language: str = Field(default="en", description="Language code")
    include_hook: bool = Field(default=True, description="Include attention-grabbing hook")
    include_cta: bool = Field(default=True, description="Include call-to-action")
    quality_preset: str = Field(default="balanced", description="Quality preset: fast, balanced, or quality")
    trending_context: Optional[Dict[str, Any]] = Field(default=None, description="Trending data context")
    channel_context: Optional[Dict[str, Any]] = Field(default=None, description="Channel style context")


class ScriptGenerationResponse(BaseModel):
    """Response model for generated script"""
    script_id: str
    title: str
    hook: str
    introduction: str
    main_content: List[Dict[str, str]]
    conclusion: str
    call_to_action: str
    timestamps: List[Dict[str, str]]
    keywords_used: List[str]
    estimated_duration: int
    word_count: int
    cost: float
    metadata: Dict[str, Any]


class ScriptVariationsRequest(BaseModel):
    """Request for generating script variations"""
    topic: str
    style: str = "informative"
    duration_minutes: int = 5
    target_audience: str = "general"
    keywords: List[str] = []
    num_variations: int = Field(default=3, ge=1, le=5)


class ScriptOptimizationRequest(BaseModel):
    """Request for optimizing an existing script"""
    script_id: str
    analytics_data: Dict[str, Any]


@router.post("/generate", response_model=ScriptGenerationResponse)
async def generate_script(
    request: ScriptGenerationRequest,
    background_tasks: BackgroundTasks,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user)
) -> ScriptGenerationResponse:
    """
    Generate a video script using AI
    
    Creates an optimized script based on the provided parameters,
    with automatic cost tracking and caching.
    """
    if not script_generator:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Script generation service not available"
        )
    
    try:
        # Create script request
        script_request = ScriptRequest(
            topic=request.topic,
            style=ScriptStyle(request.style),
            tone=ScriptTone(request.tone),
            duration_minutes=request.duration_minutes,
            target_audience=request.target_audience,
            keywords=request.keywords,
            language=request.language,
            include_hook=request.include_hook,
            include_cta=request.include_cta,
            trending_context=request.trending_context,
            channel_context=request.channel_context
        )
        
        # Generate script
        script_response = await script_generator.generate_script(
            script_request,
            quality_preset=request.quality_preset
        )
        
        # Track cost
        cost_service = CostTrackingService(db)
        await cost_service.track_cost(
            user_id=current_user.id,
            service_name="openai",
            operation="script_generation",
            cost=script_response.cost,
            metadata={
                "topic": request.topic,
                "duration": request.duration_minutes,
                "quality": request.quality_preset,
                "word_count": script_response.word_count
            }
        )
        
        # Generate script ID
        import uuid
        script_id = str(uuid.uuid4())
        
        # Store script in database (background task)
        background_tasks.add_task(
            store_script_in_db,
            db,
            script_id,
            current_user.id,
            script_response,
            request
        )
        
        return ScriptGenerationResponse(
            script_id=script_id,
            title=script_response.title,
            hook=script_response.hook,
            introduction=script_response.introduction,
            main_content=script_response.main_content,
            conclusion=script_response.conclusion,
            call_to_action=script_response.call_to_action,
            timestamps=script_response.timestamps,
            keywords_used=script_response.keywords_used,
            estimated_duration=script_response.estimated_duration,
            word_count=script_response.word_count,
            cost=script_response.cost,
            metadata=script_response.metadata
        )
        
    except Exception as e:
        logger.error(f"Script generation failed: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Script generation failed: {str(e)}"
        )


@router.post("/generate-variations")
async def generate_script_variations(
    request: ScriptVariationsRequest,
    background_tasks: BackgroundTasks,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user)
) -> Dict[str, Any]:
    """
    Generate multiple script variations for A/B testing
    
    Creates different versions of a script with varying tones
    to test which performs best.
    """
    if not script_generator:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Script generation service not available"
        )
    
    try:
        # Create base request
        base_request = ScriptRequest(
            topic=request.topic,
            style=ScriptStyle(request.style),
            tone=ScriptTone.PROFESSIONAL,  # Will be varied
            duration_minutes=request.duration_minutes,
            target_audience=request.target_audience,
            keywords=request.keywords,
            language="en",
            include_hook=True,
            include_cta=True
        )
        
        # Generate variations
        variations = await script_generator.generate_variations(
            base_request,
            num_variations=request.num_variations
        )
        
        # Calculate total cost
        total_cost = sum(v.cost for v in variations)
        
        # Track cost
        cost_service = CostTrackingService(db)
        await cost_service.track_cost(
            user_id=current_user.id,
            service_name="openai",
            operation="script_variations",
            cost=total_cost,
            metadata={
                "topic": request.topic,
                "num_variations": len(variations)
            }
        )
        
        # Format response
        return {
            "topic": request.topic,
            "num_variations": len(variations),
            "total_cost": total_cost,
            "variations": [
                {
                    "variation_id": v.metadata.get("variation_id"),
                    "tone": v.metadata.get("variation_tone"),
                    "title": v.title,
                    "hook": v.hook,
                    "word_count": v.word_count,
                    "cost": v.cost
                }
                for v in variations
            ]
        }
        
    except Exception as e:
        logger.error(f"Script variation generation failed: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to generate variations: {str(e)}"
        )


@router.post("/optimize")
async def optimize_script(
    request: ScriptOptimizationRequest,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user)
) -> Dict[str, Any]:
    """
    Optimize an existing script based on analytics data
    
    Uses performance data to improve hooks and CTAs
    for better engagement.
    """
    if not script_generator:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Script generation service not available"
        )
    
    try:
        # Retrieve script from database
        # This is a placeholder - implement actual DB retrieval
        script_data = await get_script_from_db(db, request.script_id, current_user.id)
        
        if not script_data:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Script not found"
            )
        
        # Reconstruct script response object
        from script_generation import ScriptResponse
        script = ScriptResponse(**script_data)
        
        # Optimize script
        optimized_script = await script_generator.optimize_for_engagement(
            script,
            request.analytics_data
        )
        
        # Track optimization cost
        optimization_cost = 0.002  # Approximate cost for optimization
        cost_service = CostTrackingService(db)
        await cost_service.track_cost(
            user_id=current_user.id,
            service_name="openai",
            operation="script_optimization",
            cost=optimization_cost,
            metadata={
                "script_id": request.script_id
            }
        )
        
        return {
            "script_id": request.script_id,
            "optimized": True,
            "changes": {
                "hook": optimized_script.hook != script.hook,
                "cta": optimized_script.call_to_action != script.call_to_action
            },
            "optimized_hook": optimized_script.hook,
            "optimized_cta": optimized_script.call_to_action,
            "cost": optimization_cost
        }
        
    except Exception as e:
        logger.error(f"Script optimization failed: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Optimization failed: {str(e)}"
        )


@router.get("/styles")
async def get_available_styles(
    current_user: User = Depends(get_current_user)
) -> Dict[str, List[str]]:
    """
    Get available script styles and tones
    """
    return {
        "styles": [style.value for style in ScriptStyle],
        "tones": [tone.value for tone in ScriptTone]
    }


@router.get("/cost-estimate")
async def estimate_script_cost(
    duration_minutes: int = 5,
    quality_preset: str = "balanced",
    num_variations: int = 1,
    current_user: User = Depends(get_current_user)
) -> Dict[str, Any]:
    """
    Estimate the cost of script generation
    
    Provides cost estimates based on duration and quality settings.
    """
    # Approximate token usage based on duration
    words_per_minute = 150
    total_words = duration_minutes * words_per_minute
    
    # Estimate tokens (roughly 1.3 tokens per word for English)
    estimated_tokens = int(total_words * 1.3)
    
    # Cost per 1K tokens by model
    cost_rates = {
        "fast": 0.003,      # GPT-3.5
        "balanced": 0.01,   # GPT-4 Turbo
        "quality": 0.03     # GPT-4
    }
    
    rate = cost_rates.get(quality_preset, 0.01)
    base_cost = (estimated_tokens / 1000) * rate
    total_cost = base_cost * num_variations
    
    return {
        "duration_minutes": duration_minutes,
        "quality_preset": quality_preset,
        "num_variations": num_variations,
        "estimated_words": total_words,
        "estimated_tokens": estimated_tokens,
        "cost_per_script": round(base_cost, 4),
        "total_cost": round(total_cost, 4),
        "model_used": {
            "fast": "gpt-3.5-turbo",
            "balanced": "gpt-4-turbo",
            "quality": "gpt-4"
        }.get(quality_preset, "gpt-4-turbo")
    }


# Helper functions
async def store_script_in_db(
    db: AsyncSession,
    script_id: str,
    user_id: str,
    script_response,
    request: ScriptGenerationRequest
):
    """Store generated script in database"""
    # This is a placeholder - implement actual DB storage
    # You would create a Script model and store it
    pass


async def get_script_from_db(
    db: AsyncSession,
    script_id: str,
    user_id: str
) -> Optional[Dict]:
    """Retrieve script from database"""
    # This is a placeholder - implement actual DB retrieval
    # You would query the Script model
    return None