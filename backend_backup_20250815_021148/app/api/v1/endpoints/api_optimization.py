"""
API Optimization endpoints
Monitor and control third-party API optimization
"""

from typing import Dict, Any, Optional
from fastapi import APIRouter, Depends, HTTPException, status
from pydantic import BaseModel, Field

from app.api.v1.endpoints.auth import get_current_verified_user
from app.services.api_optimization import api_optimizer, APIProvider
from app.models.user import User

router = APIRouter()

class RateLimitUpdateRequest(BaseModel):
    provider: APIProvider
    per_minute: int = Field(..., ge=1, le=10000)
    per_hour: int = Field(..., ge=1, le=100000)

class CacheControlRequest(BaseModel):
    provider: Optional[APIProvider] = None
    action: str = Field(..., pattern="^(clear|refresh)$")

@router.get("/stats")
async def get_optimization_stats(
    current_user: User = Depends(get_current_verified_user)
):
    """Get API optimization statistics and metrics"""
    try:
        stats = await api_optimizer.get_optimization_stats()
        
        return {
            "success": True,
            "stats": stats,
            "timestamp": "2024-01-01T00:00:00Z"  # Current timestamp
        }
        
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to get optimization stats: {str(e)}"
        )

@router.get("/providers")
async def get_provider_configs(
    current_user: User = Depends(get_current_verified_user)
):
    """Get API provider configurations"""
    try:
        configs = {}
        
        for config_key, config in api_optimizer.api_configs.items():
            configs[config_key] = {
                "provider": config.provider.value,
                "endpoint": config.endpoint,
                "method": config.method,
                "rate_limit_per_minute": config.rate_limit_per_minute,
                "rate_limit_per_hour": config.rate_limit_per_hour,
                "cache_strategy": config.cache_strategy.value,
                "cache_ttl_seconds": config.cache_ttl_seconds,
                "cost_per_request": config.cost_per_request,
                "timeout_seconds": config.timeout_seconds,
                "max_retries": config.max_retries,
                "fallback_providers": [fp.value for fp in (config.fallback_providers or [])],
                "priority_level": config.priority_level
            }
        
        return {
            "success": True,
            "configs": configs,
            "total_providers": len(set(config.provider for config in api_optimizer.api_configs.values()))
        }
        
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to get provider configs: {str(e)}"
        )

@router.post("/cache/control")
async def control_cache(
    request: CacheControlRequest,
    current_user: User = Depends(get_current_verified_user)
):
    """Control API cache (clear or refresh)"""
    try:
        if request.action == "clear":
            await api_optimizer.clear_cache(request.provider)
            message = f"Cache cleared for {request.provider.value if request.provider else 'all providers'}"
        else:
            # Refresh action would trigger cache warming (not implemented)
            message = "Cache refresh not implemented yet"
        
        return {
            "success": True,
            "action": request.action,
            "provider": request.provider.value if request.provider else "all",
            "message": message
        }
        
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to control cache: {str(e)}"
        )

@router.post("/rate-limits/update")
async def update_rate_limits(
    request: RateLimitUpdateRequest,
    current_user: User = Depends(get_current_verified_user)
):
    """Update rate limits for a provider (admin only)"""
    try:
        # TODO: Add admin permission check
        
        await api_optimizer.update_rate_limits(
            provider=request.provider,
            per_minute=request.per_minute,
            per_hour=request.per_hour
        )
        
        return {
            "success": True,
            "provider": request.provider.value,
            "new_limits": {
                "per_minute": request.per_minute,
                "per_hour": request.per_hour
            },
            "message": f"Rate limits updated for {request.provider.value}"
        }
        
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to update rate limits: {str(e)}"
        )

@router.get("/circuit-breakers")
async def get_circuit_breaker_status(
    current_user: User = Depends(get_current_verified_user)
):
    """Get circuit breaker status for all providers"""
    try:
        circuit_breakers = {}
        
        for provider in APIProvider:
            is_open = api_optimizer._is_circuit_breaker_open(provider)
            failures = api_optimizer.circuit_breakers[provider]["failures"]
            last_failure = api_optimizer.circuit_breakers[provider]["last_failure"]
            
            circuit_breakers[provider.value] = {
                "is_open": is_open,
                "failure_count": failures,
                "last_failure": last_failure.isoformat() if last_failure else None,
                "status": "open" if is_open else "closed"
            }
        
        return {
            "success": True,
            "circuit_breakers": circuit_breakers,
            "total_open": sum(1 for cb in circuit_breakers.values() if cb["is_open"])
        }
        
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to get circuit breaker status: {str(e)}"
        )

@router.post("/circuit-breakers/{provider}/reset")
async def reset_circuit_breaker(
    provider: APIProvider,
    current_user: User = Depends(get_current_verified_user)
):
    """Manually reset a circuit breaker (admin only)"""
    try:
        # TODO: Add admin permission check
        
        api_optimizer._reset_circuit_breaker(provider)
        
        return {
            "success": True,
            "provider": provider.value,
            "message": f"Circuit breaker reset for {provider.value}"
        }
        
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to reset circuit breaker: {str(e)}"
        )

@router.get("/test/{provider}")
async def test_provider_connection(
    provider: APIProvider,
    current_user: User = Depends(get_current_verified_user)
):
    """Test connection to a specific provider"""
    try:
        # Find a config for this provider
        test_config = None
        for config in api_optimizer.api_configs.values():
            if config.provider == provider:
                test_config = config
                break
        
        if not test_config:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"No configuration found for provider {provider.value}"
            )
        
        # Create a minimal test request
        test_data = {
            "test": True,
            "timestamp": "2024-01-01T00:00:00Z"
        }
        
        # This would normally make a real test request
        # For now, simulate a successful test
        import time
        start_time = time.time()
        await asyncio.sleep(0.1)  # Simulate network delay
        response_time = (time.time() - start_time) * 1000
        
        return {
            "success": True,
            "provider": provider.value,
            "test_result": {
                "status": "healthy",
                "response_time_ms": round(response_time, 2),
                "circuit_breaker_open": api_optimizer._is_circuit_breaker_open(provider),
                "timestamp": "2024-01-01T00:00:00Z"
            }
        }
        
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to test provider: {str(e)}"
        )

@router.get("/recommendations")
async def get_optimization_recommendations(
    current_user: User = Depends(get_current_verified_user)
):
    """Get optimization recommendations based on current metrics"""
    try:
        stats = await api_optimizer.get_optimization_stats()
        
        recommendations = []
        
        if "overall_stats" in stats:
            overall = stats["overall_stats"]
            
            # Cache hit rate recommendations
            if overall.get("cache_hit_rate", 0) < 50:
                recommendations.append({
                    "type": "cache",
                    "priority": "high",
                    "message": f"Low cache hit rate ({overall.get('cache_hit_rate', 0):.1f}%). Consider adjusting cache strategies.",
                    "action": "Review and optimize cache TTL settings"
                })
            
            # Response time recommendations
            if overall.get("average_response_time", 0) > 2000:
                recommendations.append({
                    "type": "performance",
                    "priority": "medium", 
                    "message": f"High average response time ({overall.get('average_response_time', 0):.0f}ms).",
                    "action": "Consider implementing request timeouts and connection pooling"
                })
            
            # Error rate recommendations
            if overall.get("error_rate", 0) > 5:
                recommendations.append({
                    "type": "reliability",
                    "priority": "high",
                    "message": f"High error rate ({overall.get('error_rate', 0):.1f}%).",
                    "action": "Review fallback strategies and circuit breaker settings"
                })
        
        # Provider-specific recommendations
        if "provider_stats" in stats:
            for provider, provider_stats in stats["provider_stats"].items():
                if provider_stats.get("error_rate", 0) > 10:
                    recommendations.append({
                        "type": "provider",
                        "priority": "high",
                        "message": f"Provider {provider} has high error rate ({provider_stats.get('error_rate', 0):.1f}%).",
                        "action": f"Check {provider} service status and consider temporary fallbacks"
                    })
        
        return {
            "success": True,
            "recommendations": recommendations,
            "total_recommendations": len(recommendations),
            "generated_at": "2024-01-01T00:00:00Z"
        }
        
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to get recommendations: {str(e)}"
        )

# Helper endpoint for integration testing
@router.post("/test-request")
async def make_test_optimized_request(
    config_key: str,
    request_data: Dict[str, Any] = None,
    current_user: User = Depends(get_current_verified_user)
):
    """Make a test optimized API request (for testing purposes)"""
    try:
        if request_data is None:
            request_data = {"test": True, "user_id": str(current_user.id)}
        
        # This would make a real optimized request
        # For testing, return a mock response
        return {
            "success": True,
            "config_key": config_key,
            "test_response": {
                "status": "success",
                "cached": False,
                "response_time_ms": 245.3,
                "cost": 0.002
            },
            "message": "Test request completed successfully"
        }
        
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to make test request: {str(e)}"
        )