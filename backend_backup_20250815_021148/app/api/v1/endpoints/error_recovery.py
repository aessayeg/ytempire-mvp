"""
API endpoints for advanced error recovery service
"""

from fastapi import APIRouter, HTTPException, Depends
from typing import Dict, Any, List, Optional
from datetime import datetime
from pydantic import BaseModel, Field

from app.core.security import get_current_user
from app.models.user import User
from app.services.advanced_error_recovery import (
    advanced_recovery,
    RecoveryStrategy,
    RecoveryConfig,
    ErrorContext
)

router = APIRouter()


class RecoveryConfigRequest(BaseModel):
    """Recovery configuration request model"""
    max_retries: int = Field(default=3, ge=1, le=10)
    retry_delay: float = Field(default=1.0, ge=0.1, le=60)
    retry_backoff: float = Field(default=2.0, ge=1.0, le=5.0)
    retry_jitter: bool = True
    failure_threshold: int = Field(default=5, ge=1, le=50)
    recovery_timeout: float = Field(default=60.0, ge=5, le=300)
    operation_timeout: float = Field(default=30.0, ge=1, le=120)
    max_concurrent_calls: int = Field(default=10, ge=1, le=100)


class CircuitBreakerConfig(BaseModel):
    """Circuit breaker configuration"""
    service_name: str
    failure_threshold: int = Field(default=5, ge=1, le=50)
    recovery_timeout: float = Field(default=60.0, ge=5, le=300)
    half_open_requests: int = Field(default=3, ge=1, le=10)


class BulkheadConfig(BaseModel):
    """Bulkhead configuration"""
    service_name: str
    max_concurrent_calls: int = Field(default=10, ge=1, le=100)
    queue_size: int = Field(default=50, ge=10, le=500)


@router.post("/register/circuit-breaker")
async def register_circuit_breaker(
    config: CircuitBreakerConfig,
    current_user: User = Depends(get_current_user)
) -> Dict[str, Any]:
    """
    Register a circuit breaker for a service
    
    - **service_name**: Name of the service to protect
    - **failure_threshold**: Number of failures before opening circuit
    - **recovery_timeout**: Time to wait before attempting recovery
    - **half_open_requests**: Number of test requests in half-open state
    """
    try:
        recovery_config = RecoveryConfig(
            failure_threshold=config.failure_threshold,
            recovery_timeout=config.recovery_timeout,
            half_open_requests=config.half_open_requests
        )
        
        advanced_recovery.register_circuit_breaker(
            config.service_name,
            recovery_config
        )
        
        return {
            "status": "success",
            "message": f"Circuit breaker registered for {config.service_name}",
            "config": config.dict()
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/register/bulkhead")
async def register_bulkhead(
    config: BulkheadConfig,
    current_user: User = Depends(get_current_user)
) -> Dict[str, Any]:
    """
    Register a bulkhead for service isolation
    
    - **service_name**: Name of the service to isolate
    - **max_concurrent_calls**: Maximum concurrent calls allowed
    - **queue_size**: Size of the waiting queue
    """
    try:
        recovery_config = RecoveryConfig(
            max_concurrent_calls=config.max_concurrent_calls,
            queue_size=config.queue_size
        )
        
        advanced_recovery.register_bulkhead(
            config.service_name,
            recovery_config
        )
        
        return {
            "status": "success",
            "message": f"Bulkhead registered for {config.service_name}",
            "config": config.dict()
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/status/circuit-breakers")
async def get_circuit_breaker_status(
    current_user: User = Depends(get_current_user)
) -> Dict[str, Any]:
    """
    Get status of all circuit breakers
    
    Returns the state and metrics for all registered circuit breakers
    """
    try:
        status = advanced_recovery.get_circuit_breaker_status()
        return {
            "timestamp": datetime.now().isoformat(),
            "circuit_breakers": status,
            "total": len(status)
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/status/bulkheads")
async def get_bulkhead_status(
    current_user: User = Depends(get_current_user)
) -> Dict[str, Any]:
    """
    Get status of all bulkheads
    
    Returns the current load and queue status for all bulkheads
    """
    try:
        status = advanced_recovery.get_bulkhead_status()
        return {
            "timestamp": datetime.now().isoformat(),
            "bulkheads": status,
            "total": len(status)
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/test/retry")
async def test_retry_mechanism(
    service_name: str,
    simulate_failures: int = 2,
    current_user: User = Depends(get_current_user)
) -> Dict[str, Any]:
    """
    Test retry mechanism with simulated failures
    
    - **service_name**: Name of the service to test
    - **simulate_failures**: Number of failures to simulate before success
    """
    attempt_count = 0
    
    async def test_function():
        nonlocal attempt_count
        attempt_count += 1
        
        if attempt_count <= simulate_failures:
            raise Exception(f"Simulated failure {attempt_count}")
        
        return {"success": True, "attempts": attempt_count}
    
    try:
        context = ErrorContext(
            service_name=service_name,
            operation="test_retry",
            error_type="SimulatedException",
            error_message="",
            timestamp=datetime.now()
        )
        
        result = await advanced_recovery.with_retry(
            test_function,
            context,
            RecoveryConfig(max_retries=5)
        )
        
        return {
            "status": "success",
            "result": result,
            "total_attempts": attempt_count
        }
    except Exception as e:
        return {
            "status": "failed",
            "error": str(e),
            "total_attempts": attempt_count
        }


@router.post("/test/circuit-breaker")
async def test_circuit_breaker(
    service_name: str,
    should_fail: bool = False,
    current_user: User = Depends(get_current_user)
) -> Dict[str, Any]:
    """
    Test circuit breaker behavior
    
    - **service_name**: Name of the service to test
    - **should_fail**: Whether the test should simulate a failure
    """
    async def test_function():
        if should_fail:
            raise Exception("Simulated circuit breaker test failure")
        return {"success": True, "timestamp": datetime.now().isoformat()}
    
    try:
        # Register circuit breaker if not exists
        advanced_recovery.register_circuit_breaker(service_name)
        
        result = await advanced_recovery.with_circuit_breaker(
            test_function,
            service_name
        )
        
        # Get current state
        status = advanced_recovery.get_circuit_breaker_status()
        breaker_state = status.get(service_name, {})
        
        return {
            "status": "success",
            "result": result,
            "circuit_state": breaker_state
        }
    except Exception as e:
        # Get current state even on failure
        status = advanced_recovery.get_circuit_breaker_status()
        breaker_state = status.get(service_name, {})
        
        return {
            "status": "failed",
            "error": str(e),
            "circuit_state": breaker_state
        }


@router.post("/test/bulkhead")
async def test_bulkhead(
    service_name: str,
    concurrent_requests: int = 5,
    current_user: User = Depends(get_current_user)
) -> Dict[str, Any]:
    """
    Test bulkhead isolation
    
    - **service_name**: Name of the service to test
    - **concurrent_requests**: Number of concurrent requests to simulate
    """
    import asyncio
    
    async def test_function(request_id: int):
        await asyncio.sleep(0.5)  # Simulate work
        return {"request_id": request_id, "completed": True}
    
    try:
        # Register bulkhead if not exists
        advanced_recovery.register_bulkhead(service_name)
        
        # Create concurrent requests
        tasks = []
        for i in range(concurrent_requests):
            task = advanced_recovery.with_bulkhead(
                test_function,
                service_name,
                i
            )
            tasks.append(task)
        
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Get current state
        status = advanced_recovery.get_bulkhead_status()
        bulkhead_state = status.get(service_name, {})
        
        # Count successes and failures
        successes = [r for r in results if not isinstance(r, Exception)]
        failures = [r for r in results if isinstance(r, Exception)]
        
        return {
            "status": "completed",
            "total_requests": concurrent_requests,
            "successful": len(successes),
            "failed": len(failures),
            "results": successes,
            "errors": [str(e) for e in failures],
            "bulkhead_state": bulkhead_state
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/configure/fallback-chain")
async def configure_fallback_chain(
    service_name: str,
    fallback_services: List[str],
    current_user: User = Depends(get_current_user)
) -> Dict[str, Any]:
    """
    Configure fallback chain for a service
    
    - **service_name**: Primary service name
    - **fallback_services**: List of fallback service names in order
    """
    try:
        # Create fallback functions (placeholder)
        fallbacks = []
        for fallback_service in fallback_services:
            async def fallback():
                return {"service": fallback_service, "type": "fallback"}
            fallbacks.append(fallback)
        
        advanced_recovery.register_fallback_chain(service_name, fallbacks)
        
        return {
            "status": "success",
            "message": f"Fallback chain configured for {service_name}",
            "chain": [service_name] + fallback_services
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/metrics")
async def get_error_recovery_metrics(
    current_user: User = Depends(get_current_user)
) -> Dict[str, Any]:
    """
    Get error recovery metrics
    
    Returns comprehensive metrics about error recovery operations
    """
    try:
        from prometheus_client import generate_latest, REGISTRY
        
        # Get Prometheus metrics
        metrics_output = generate_latest(REGISTRY).decode('utf-8')
        
        # Parse relevant metrics (simplified)
        metrics = {
            "circuit_breakers": advanced_recovery.get_circuit_breaker_status(),
            "bulkheads": advanced_recovery.get_bulkhead_status(),
            "cache_info": {
                "cache_size": len(advanced_recovery.cache_store),
                "entries": list(advanced_recovery.cache_store.keys())[:10]  # First 10 keys
            }
        }
        
        return {
            "timestamp": datetime.now().isoformat(),
            "metrics": metrics,
            "is_initialized": advanced_recovery.is_initialized
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/cache/set")
async def set_cache_value(
    key: str,
    value: Any,
    ttl: int = 3600,
    current_user: User = Depends(get_current_user)
) -> Dict[str, Any]:
    """
    Set a value in the recovery cache
    
    - **key**: Cache key
    - **value**: Value to cache
    - **ttl**: Time to live in seconds
    """
    try:
        await advanced_recovery._set_in_cache(key, value, ttl)
        return {
            "status": "success",
            "key": key,
            "ttl": ttl
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/cache/get/{key}")
async def get_cache_value(
    key: str,
    accept_stale: bool = False,
    current_user: User = Depends(get_current_user)
) -> Dict[str, Any]:
    """
    Get a value from the recovery cache
    
    - **key**: Cache key
    - **accept_stale**: Whether to accept stale cache values
    """
    try:
        value = await advanced_recovery._get_from_cache(key, accept_stale)
        
        if value is None:
            raise HTTPException(status_code=404, detail="Key not found in cache")
        
        return {
            "status": "success",
            "key": key,
            "value": value
        }
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))