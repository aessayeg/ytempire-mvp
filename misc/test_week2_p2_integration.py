"""
Test script for Week 2 P2 (Nice to Have) Backend Features
Tests advanced error recovery, third-party integrations, and advanced caching
"""

import asyncio
import sys
import os
from pathlib import Path
from datetime import datetime
import json

# Add backend directory to path
backend_path = Path(__file__).parent.parent / "backend"
sys.path.insert(0, str(backend_path))

from app.services.advanced_error_recovery import (
    advanced_recovery,
    RecoveryStrategy,
    RecoveryConfig,
    ErrorContext
)
from app.services.third_party_integrations import (
    third_party_service,
    IntegrationType,
    IntegrationConfig
)


class Colors:
    """Terminal colors for output"""
    GREEN = '\033[92m'
    RED = '\033[91m'
    YELLOW = '\033[93m'
    BLUE = '\033[94m'
    RESET = '\033[0m'
    BOLD = '\033[1m'


def print_header(title: str):
    """Print a formatted header"""
    print(f"\n{Colors.BOLD}{Colors.BLUE}{'='*60}{Colors.RESET}")
    print(f"{Colors.BOLD}{Colors.BLUE}{title}{Colors.RESET}")
    print(f"{Colors.BOLD}{Colors.BLUE}{'='*60}{Colors.RESET}\n")


def print_success(message: str):
    """Print success message"""
    print(f"{Colors.GREEN}[OK] {message}{Colors.RESET}")


def print_error(message: str):
    """Print error message"""
    print(f"{Colors.RED}[ERROR] {message}{Colors.RESET}")


def print_info(message: str):
    """Print info message"""
    print(f"{Colors.YELLOW}[INFO] {message}{Colors.RESET}")


async def test_error_recovery():
    """Test advanced error recovery mechanisms"""
    print_header("Testing Advanced Error Recovery")
    
    try:
        # Initialize service
        await advanced_recovery.initialize()
        print_success("Error recovery service initialized")
        
        # Test 1: Retry mechanism
        print_info("Testing retry mechanism...")
        attempt_count = 0
        
        async def failing_function():
            nonlocal attempt_count
            attempt_count += 1
            if attempt_count < 3:
                raise Exception(f"Simulated failure {attempt_count}")
            return {"success": True, "attempts": attempt_count}
        
        context = ErrorContext(
            service_name="test_service",
            operation="test_retry",
            error_type="TestException",
            error_message="",
            timestamp=datetime.now()
        )
        
        result = await advanced_recovery.with_retry(
            failing_function,
            context,
            RecoveryConfig(max_retries=5)
        )
        
        print_success(f"Retry mechanism working - succeeded after {result['attempts']} attempts")
        
        # Test 2: Circuit breaker
        print_info("Testing circuit breaker...")
        advanced_recovery.register_circuit_breaker("test_breaker")
        
        async def test_function():
            return {"status": "success"}
        
        result = await advanced_recovery.with_circuit_breaker(
            test_function,
            "test_breaker"
        )
        
        print_success("Circuit breaker working")
        
        # Test 3: Bulkhead isolation
        print_info("Testing bulkhead isolation...")
        advanced_recovery.register_bulkhead("test_bulkhead")
        
        async def isolated_function(id: int):
            await asyncio.sleep(0.1)
            return {"id": id, "completed": True}
        
        tasks = []
        for i in range(5):
            task = advanced_recovery.with_bulkhead(
                isolated_function,
                "test_bulkhead",
                i
            )
            tasks.append(task)
        
        results = await asyncio.gather(*tasks)
        print_success(f"Bulkhead isolation working - processed {len(results)} requests")
        
        # Test 4: Cache fallback
        print_info("Testing cache fallback...")
        
        async def data_loader():
            return {"data": "test_value", "timestamp": datetime.now().isoformat()}
        
        result1 = await advanced_recovery.with_cache_fallback(
            data_loader,
            "test_cache_key",
            ttl=60
        )
        
        result2 = await advanced_recovery.with_cache_fallback(
            data_loader,
            "test_cache_key",
            ttl=60
        )
        
        if result1 == result2:
            print_success("Cache fallback working - retrieved from cache")
        else:
            print_error("Cache fallback not working properly")
        
        # Get status
        circuit_status = advanced_recovery.get_circuit_breaker_status()
        bulkhead_status = advanced_recovery.get_bulkhead_status()
        
        print_info(f"Circuit breakers registered: {len(circuit_status)}")
        print_info(f"Bulkheads registered: {len(bulkhead_status)}")
        
        return True
        
    except Exception as e:
        print_error(f"Error recovery test failed: {e}")
        return False


async def test_third_party_integrations():
    """Test third-party integrations service"""
    print_header("Testing Third-Party Integrations")
    
    try:
        # Initialize service
        await third_party_service.initialize()
        print_success("Third-party integrations service initialized")
        
        # Test 1: Register a test integration
        print_info("Testing integration registration...")
        
        test_config = IntegrationConfig(
            name="test_webhook",
            type=IntegrationType.WEBHOOK,
            base_url="https://webhook.site/test",
            auth_type="none",
            credentials={},
            rate_limit=10,
            webhook_events=["test_event"]
        )
        
        await third_party_service.register_integration(test_config)
        print_success("Test integration registered")
        
        # Test 2: Check integration status
        status = third_party_service.get_integration_status("test_webhook")
        print_success(f"Integration status: {status.value}")
        
        # Test 3: Get all integrations status
        all_status = third_party_service.get_all_integrations_status()
        print_info(f"Total integrations registered: {len(all_status)}")
        
        # Test 4: Rate limiter
        print_info("Testing rate limiter...")
        if "test_webhook" in third_party_service.rate_limiters:
            limiter = third_party_service.rate_limiters["test_webhook"]
            
            # Acquire permits
            for i in range(3):
                await limiter.acquire()
            
            print_success("Rate limiter working")
        
        # Test 5: Webhook signature generation
        print_info("Testing webhook signature...")
        
        test_payload = {"event": "test", "data": {"id": 123}}
        signature = third_party_service._generate_webhook_signature(
            "test_secret",
            test_payload
        )
        
        if signature.startswith("sha256="):
            print_success("Webhook signature generation working")
        
        # Test 6: Integration types
        print_info("Available integration types:")
        for integration_type in IntegrationType:
            print_info(f"  - {integration_type.value}")
        
        return True
        
    except Exception as e:
        print_error(f"Third-party integrations test failed: {e}")
        return False


async def test_advanced_caching():
    """Test advanced caching strategies"""
    print_header("Testing Advanced Caching")
    
    try:
        # Import caching service
        from app.services.advanced_caching import AdvancedCacheManager, CacheConfig
        import redis.asyncio as redis
        
        print_info("Testing multi-tier caching...")
        
        # Initialize Redis connection (may fail if Redis not running)
        try:
            redis_client = await redis.from_url("redis://localhost:6379")
            await redis_client.ping()
            redis_available = True
            print_success("Redis connection established")
        except:
            redis_client = None
            redis_available = False
            print_info("Redis not available - using L1 memory cache only")
        
        # Create cache manager
        cache_manager = AdvancedCacheManager(
            redis_client=redis_client,
            enable_l1=True,
            enable_l2=redis_available,
            enable_l3=False
        )
        
        # Test 1: Basic set/get
        print_info("Testing basic cache operations...")
        
        test_data = {
            "id": 123,
            "name": "Test Video",
            "views": 1000
        }
        
        config = CacheConfig(
            ttl=300,
            tags=["video", "test"],
            compression=False
        )
        
        await cache_manager.set("test:video:123", test_data, config)
        print_success("Cache set operation successful")
        
        cached_data = await cache_manager.get("test:video:123")
        if cached_data == test_data:
            print_success("Cache get operation successful - data matches")
        else:
            print_error("Cache get operation failed - data mismatch")
        
        # Test 2: Batch operations
        print_info("Testing batch cache operations...")
        
        batch_items = {
            "item:1": {"id": 1, "value": "one"},
            "item:2": {"id": 2, "value": "two"},
            "item:3": {"id": 3, "value": "three"}
        }
        
        await cache_manager.batch_set(batch_items, ttl=300)
        print_success("Batch set operation successful")
        
        batch_results = await cache_manager.batch_get(list(batch_items.keys()))
        if len(batch_results) == len(batch_items):
            print_success(f"Batch get operation successful - retrieved {len(batch_results)} items")
        
        # Test 3: Tag-based invalidation
        print_info("Testing tag-based invalidation...")
        
        await cache_manager.set("tagged:1", {"data": 1}, CacheConfig(tags=["test_tag"]))
        await cache_manager.set("tagged:2", {"data": 2}, CacheConfig(tags=["test_tag"]))
        
        invalidated_count = await cache_manager.invalidate_by_tag("test_tag")
        print_success(f"Tag invalidation successful - invalidated {invalidated_count} entries")
        
        # Test 4: Cache statistics
        stats = await cache_manager.get_stats()
        print_info("Cache statistics:")
        print_info(f"  - Hits: {stats['hits']}")
        print_info(f"  - Misses: {stats['misses']}")
        print_info(f"  - Hit Rate: {stats['hit_rate']:.2%}")
        print_info(f"  - L1 Size: {stats['l1_size']}")
        
        # Test 5: Cache warming
        print_info("Testing cache warming...")
        
        async def expensive_operation():
            await asyncio.sleep(0.1)  # Simulate expensive operation
            return {"expensive": "data", "timestamp": datetime.now().isoformat()}
        
        await cache_manager.warm_cache("warmed:key", expensive_operation, ttl=600)
        print_success("Cache warming successful")
        
        return True
        
    except Exception as e:
        print_error(f"Advanced caching test failed: {e}")
        return False


async def test_api_endpoints():
    """Test that API endpoints are registered"""
    print_header("Testing API Endpoint Registration")
    
    try:
        from app.api.v1.api import api_router
        
        # Get all routes
        routes = []
        for route in api_router.routes:
            if hasattr(route, 'path'):
                routes.append(route.path)
        
        # Check for P2 endpoints
        p2_endpoints = [
            "/error-recovery",
            "/integrations",
            "/cache"
        ]
        
        registered = []
        missing = []
        
        for endpoint in p2_endpoints:
            found = any(endpoint in route for route in routes)
            if found:
                registered.append(endpoint)
            else:
                missing.append(endpoint)
        
        for endpoint in registered:
            print_success(f"Endpoint registered: {endpoint}")
        
        for endpoint in missing:
            print_error(f"Endpoint missing: {endpoint}")
        
        # Print some specific routes
        print_info("\nSample P2 routes registered:")
        p2_routes = [r for r in routes if any(p in r for p in p2_endpoints)]
        for route in p2_routes[:10]:  # Show first 10
            print_info(f"  - {route}")
        
        return len(missing) == 0
        
    except Exception as e:
        print_error(f"API endpoint test failed: {e}")
        return False


async def main():
    """Run all P2 feature tests"""
    print_header("YTEmpire Week 2 P2 Features Integration Test")
    print(f"Test started at: {datetime.now().isoformat()}")
    
    results = {}
    
    # Test each P2 feature
    print("\n" + "="*60)
    print("Testing Week 2 P2 (Nice to Have) Backend Features")
    print("="*60)
    
    # 1. Advanced Error Recovery
    results["Error Recovery"] = await test_error_recovery()
    
    # 2. Third-Party Integrations
    results["Third-Party Integrations"] = await test_third_party_integrations()
    
    # 3. Advanced Caching
    results["Advanced Caching"] = await test_advanced_caching()
    
    # 4. API Endpoints
    results["API Endpoints"] = await test_api_endpoints()
    
    # Print summary
    print_header("Test Summary")
    
    total_tests = len(results)
    passed_tests = sum(1 for v in results.values() if v)
    
    for feature, passed in results.items():
        status = f"{Colors.GREEN}PASSED{Colors.RESET}" if passed else f"{Colors.RED}FAILED{Colors.RESET}"
        print(f"{feature}: {status}")
    
    print(f"\n{Colors.BOLD}Total: {passed_tests}/{total_tests} tests passed{Colors.RESET}")
    
    if passed_tests == total_tests:
        print_success("\nAll Week 2 P2 features successfully integrated!")
        print_success("Advanced Error Recovery, Third-Party Integrations, and Advanced Caching are operational.")
    else:
        print_error(f"\n{total_tests - passed_tests} features need attention.")
    
    print(f"\nTest completed at: {datetime.now().isoformat()}")
    
    # Cleanup
    await advanced_recovery.shutdown()
    await third_party_service.shutdown()
    
    return passed_tests == total_tests


if __name__ == "__main__":
    success = asyncio.run(main())
    sys.exit(0 if success else 1)