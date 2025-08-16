"""
Cost Optimization API Endpoints
Manages AI/ML cost optimization strategies
"""
from typing import Dict, Any, List, Optional
from datetime import datetime, timedelta
from fastapi import APIRouter, Depends, HTTPException, status, Query
from sqlalchemy.ext.asyncio import AsyncSession
from pydantic import BaseModel, Field

from app.db.session import get_async_db
from app.api.v1.endpoints.auth import get_current_user, get_current_verified_user
from app.models.user import User
from app.services.cost_optimizer import (
    get_cost_optimizer,
    ServiceType,
    ModelTier,
    OptimizationStrategy,
)

router = APIRouter()


class CostEstimateRequest(BaseModel):
    """Request for cost estimation"""

    service: str = Field(
        ..., description="Service type (script_generation, voice_synthesis, etc.)"
    )
    model: str = Field(default="gpt-3.5-turbo", description="AI model to use")
    input_tokens: int = Field(..., ge=1, description="Estimated input tokens")
    output_tokens: Optional[int] = Field(None, description="Estimated output tokens")


class OptimalModelRequest(BaseModel):
    """Request for optimal model selection"""

    service: str = Field(..., description="Service type")
    quality_required: float = Field(
        default=85.0, ge=0, le=100, description="Required quality score"
    )
    budget_remaining: Optional[float] = Field(
        None, description="Remaining budget for this video"
    )


class CostValidationResponse(BaseModel):
    """Cost validation response"""

    video_id: int
    total_cost: float
    target: float
    within_target: bool
    overage: float
    savings_achieved: float


class CostReportResponse(BaseModel):
    """Cost optimization report"""

    period_days: int
    total_cost: float
    total_videos: int
    cost_per_video: float
    target_cost_per_video: float
    within_target: bool
    costs_by_service: Dict[str, Any]
    optimization_opportunities: List[Dict[str, str]]
    active_strategies: List[str]
    estimated_savings: float


class OptimizationStrategyResponse(BaseModel):
    """Optimization strategy details"""

    name: str
    description: str
    savings_percentage: float
    quality_impact: float
    implementation_complexity: str


@router.post("/estimate")
async def estimate_cost(
    request: CostEstimateRequest,
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_async_db),
):
    """Estimate cost for an AI service call"""
    try:
        optimizer = get_cost_optimizer()

        # Validate service type
        try:
            service = ServiceType(request.service)
        except ValueError:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Invalid service type: {request.service}",
            )

        # Estimate cost
        estimated_cost = await optimizer.estimate_cost(
            service=service,
            model=request.model,
            input_tokens=request.input_tokens,
            output_tokens=request.output_tokens,
        )

        # Check if budget available
        budget_available = await optimizer.check_budget_available(
            service=service, estimated_cost=estimated_cost
        )

        return {
            "service": request.service,
            "model": request.model,
            "estimated_cost": estimated_cost,
            "budget_available": budget_available,
            "service_limit": optimizer.COST_LIMITS.get(service, 1.0),
            "daily_spent": await optimizer.get_daily_spending(),
            "daily_limit": optimizer.DAILY_BUDGET["total"],
        }

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Cost estimation failed: {str(e)}",
        )


@router.post("/optimal-model")
async def get_optimal_model(
    request: OptimalModelRequest,
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_async_db),
):
    """Get optimal AI model based on budget and quality requirements"""
    try:
        optimizer = get_cost_optimizer()

        # Validate service type
        try:
            service = ServiceType(request.service)
        except ValueError:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Invalid service type: {request.service}",
            )

        # Get optimal model
        model, tier = await optimizer.get_optimal_model(
            service=service,
            quality_required=request.quality_required,
            budget_remaining=request.budget_remaining,
        )

        # Estimate cost for this model
        estimated_cost = await optimizer.estimate_cost(
            service=service,
            model=model,
            input_tokens=1000,  # Average estimate
            output_tokens=2000,
        )

        return {
            "service": request.service,
            "recommended_model": model,
            "tier": tier.value,
            "quality_score": optimizer._estimate_model_quality(model, service),
            "estimated_cost": estimated_cost,
            "quality_required": request.quality_required,
            "meets_requirements": optimizer._estimate_model_quality(model, service)
            >= request.quality_required,
        }

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Model selection failed: {str(e)}",
        )


@router.get("/report", response_model=CostReportResponse)
async def get_cost_report(
    days: int = Query(default=7, ge=1, le=30, description="Number of days for report"),
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_async_db),
):
    """Generate cost optimization report"""
    try:
        optimizer = get_cost_optimizer()
        report = await optimizer.get_cost_report(days=days)

        return CostReportResponse(**report)

    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Report generation failed: {str(e)}",
        )


@router.get("/validate/{video_id}", response_model=CostValidationResponse)
async def validate_video_cost(
    video_id: int,
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_async_db),
):
    """Validate if a video met the <$3 cost target"""
    try:
        optimizer = get_cost_optimizer()
        validation = await optimizer.validate_video_cost(video_id)

        return CostValidationResponse(**validation)

    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Validation failed: {str(e)}",
        )


@router.get("/strategies", response_model=List[OptimizationStrategyResponse])
async def list_optimization_strategies(
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_async_db),
):
    """List available optimization strategies"""
    try:
        optimizer = get_cost_optimizer()
        strategies = optimizer.optimization_strategies

        return [
            OptimizationStrategyResponse(
                name=s.name,
                description=s.description,
                savings_percentage=s.savings_percentage,
                quality_impact=s.quality_impact,
                implementation_complexity=s.implementation_complexity,
            )
            for s in strategies
        ]

    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to list strategies: {str(e)}",
        )


@router.get("/daily-spending")
async def get_daily_spending(
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_async_db),
):
    """Get current daily spending status"""
    try:
        optimizer = get_cost_optimizer()
        daily_spent = await optimizer.get_daily_spending()

        return {
            "date": datetime.utcnow().date().isoformat(),
            "total_spent": daily_spent,
            "daily_limit": optimizer.DAILY_BUDGET["total"],
            "remaining": optimizer.DAILY_BUDGET["total"] - daily_spent,
            "percentage_used": (daily_spent / optimizer.DAILY_BUDGET["total"]) * 100,
            "budget_breakdown": {
                service: {"limit": limit, "status": "active"}
                for service, limit in optimizer.DAILY_BUDGET.items()
                if service != "total"
            },
        }

    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to get spending data: {str(e)}",
        )


@router.post("/optimize-prompt")
async def optimize_prompt(
    prompt: str = Query(..., description="Prompt to optimize"),
    max_tokens: Optional[int] = Query(None, description="Maximum tokens allowed"),
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_async_db),
):
    """Optimize a prompt to reduce token usage"""
    try:
        optimizer = get_cost_optimizer()
        optimized = await optimizer.optimize_prompt(prompt, max_tokens)

        # Calculate token savings (rough estimate)
        original_tokens = len(prompt.split()) * 1.3  # Rough token estimate
        optimized_tokens = len(optimized.split()) * 1.3

        return {
            "original_prompt": prompt,
            "optimized_prompt": optimized,
            "original_tokens": int(original_tokens),
            "optimized_tokens": int(optimized_tokens),
            "tokens_saved": int(original_tokens - optimized_tokens),
            "savings_percentage": (
                (original_tokens - optimized_tokens) / original_tokens
            )
            * 100
            if original_tokens > 0
            else 0,
        }

    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Prompt optimization failed: {str(e)}",
        )


@router.get("/cache-stats")
async def get_cache_statistics(
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_async_db),
):
    """Get cache statistics for cost optimization"""
    try:
        optimizer = get_cost_optimizer()

        # Get cache keys
        cache_keys = optimizer.redis.keys("ai_cache:*")

        # Calculate cache stats
        cache_by_service = {}
        for key in cache_keys:
            service = key.split(":")[1] if ":" in key else "unknown"
            if service not in cache_by_service:
                cache_by_service[service] = 0
            cache_by_service[service] += 1

        # Estimate savings from caching
        estimated_savings = len(cache_keys) * 0.05  # Assume $0.05 saved per cache hit

        return {
            "total_cached_responses": len(cache_keys),
            "cache_by_service": cache_by_service,
            "estimated_savings": estimated_savings,
            "cache_ttl_settings": {
                service.value: ttl for service, ttl in optimizer.CACHE_TTL.items()
            },
        }

    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to get cache stats: {str(e)}",
        )
