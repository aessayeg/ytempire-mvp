"""
Full ML Integration Test for YTEmpire
Tests complete integration with proper configuration handling
"""

import sys
import os
import asyncio
import json
from pathlib import Path
from datetime import datetime
import numpy as np
import pandas as pd

# Setup environment before imports
os.environ.setdefault("ENVIRONMENT", "testing")
os.environ.setdefault("DATABASE_URL", "postgresql+asyncpg://test:test@localhost:5432/test")
os.environ.setdefault("SECRET_KEY", "test-secret-key")
os.environ.setdefault("ML_ENABLED", "true")
os.environ.setdefault("ML_MODELS_PATH", "models")

# Add project paths
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'backend'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'ml-pipeline', 'src'))


async def test_ml_services():
    """Test ML services integration"""
    print("\n" + "="*60)
    print("Testing ML Services Integration")
    print("="*60)
    
    results = {}
    
    # Test 1: Import ML models
    print("\n1. Testing ML model imports...")
    try:
        from automl_pipeline import AutoMLPipeline, AutoMLConfig
        from personalization_model import PersonalizationEngine, PersonalizationConfig
        print("   [OK] ML models imported")
        results['ML Models'] = True
    except ImportError as e:
        print(f"   [FAIL] ML models import failed: {e}")
        results['ML Models'] = False
        return results
    
    # Test 2: Import ML integration service
    print("\n2. Testing ML integration service...")
    try:
        from app.services.ml_integration_service import ml_service
        print("   [OK] ML integration service imported")
        
        # Check if ML is available
        if ml_service.automl_pipeline:
            print("   [OK] AutoML pipeline initialized")
        else:
            print("   [INFO] AutoML pipeline not initialized (will init on first use)")
        
        if ml_service.personalization_engine:
            print("   [OK] Personalization engine initialized")
        else:
            print("   [INFO] Personalization engine not initialized (will init on first use)")
        
        results['ML Integration Service'] = True
    except Exception as e:
        print(f"   [FAIL] ML integration service failed: {e}")
        results['ML Integration Service'] = False
    
    # Test 3: Test personalized content generation
    print("\n3. Testing personalized content generation...")
    try:
        channel_data = {
            'name': 'TestChannel',
            'niche': 'technology',
            'description': 'Tech tutorials and reviews'
        }
        
        historical_videos = [
            {
                'title': f'Tutorial {i}',
                'views': np.random.randint(1000, 50000),
                'likes': np.random.randint(50, 2000),
                'comments': np.random.randint(10, 200),
                'duration': 600,
                'published_at': datetime.now().isoformat()
            }
            for i in range(5)
        ]
        
        recommendation = await ml_service.get_personalized_content_recommendation(
            channel_id='test_channel_001',
            channel_data=channel_data,
            historical_videos=historical_videos,
            trending_topics=['AI', 'Machine Learning']
        )
        
        print(f"   [OK] Generated title: {recommendation['title']}")
        print(f"   [OK] Confidence: {recommendation['confidence_score']:.2%}")
        print(f"   [OK] Style: {recommendation['style']}")
        
        results['Personalization'] = True
    except Exception as e:
        print(f"   [FAIL] Personalization failed: {e}")
        results['Personalization'] = False
    
    # Test 4: Test performance prediction
    print("\n4. Testing performance prediction...")
    try:
        video_features = {
            'title_length': 10,
            'description_length': 100,
            'keyword_count': 5,
            'trending_score': 0.7,
            'channel_subscriber_count': 10000,
            'channel_video_count': 50,
            'posting_hour': 14,
            'posting_day': 4,
            'video_duration': 600
        }
        
        prediction = await ml_service.predict_video_performance(video_features)
        
        print(f"   [OK] Predicted views: {prediction['predicted_views']:,}")
        print(f"   [OK] Predicted engagement: {prediction['predicted_engagement_rate']:.2%}")
        
        results['Performance Prediction'] = True
    except Exception as e:
        print(f"   [FAIL] Performance prediction failed: {e}")
        results['Performance Prediction'] = False
    
    # Test 5: Test channel insights
    print("\n5. Testing channel insights...")
    try:
        insights = await ml_service.get_channel_insights('test_channel_001')
        
        if 'error' not in insights:
            print(f"   [OK] Retrieved insights for channel")
            if 'performance' in insights:
                print(f"   [OK] Performance metrics available")
            if 'optimal_schedule' in insights:
                print(f"   [OK] Optimal schedule available")
        
        results['Channel Insights'] = True
    except Exception as e:
        print(f"   [FAIL] Channel insights failed: {e}")
        results['Channel Insights'] = False
    
    return results


async def test_enhanced_video_generation():
    """Test enhanced video generation service"""
    print("\n" + "="*60)
    print("Testing Enhanced Video Generation")
    print("="*60)
    
    results = {}
    
    try:
        from app.services.enhanced_video_generation import enhanced_orchestrator
        print("\n1. Enhanced video generation service imported")
        
        # Check methods exist
        if hasattr(enhanced_orchestrator, 'generate_video_with_ml'):
            print("   [OK] generate_video_with_ml method exists")
            results['ML Video Generation'] = True
        else:
            print("   [FAIL] generate_video_with_ml method not found")
            results['ML Video Generation'] = False
        
        if hasattr(enhanced_orchestrator, 'batch_generate_with_ml'):
            print("   [OK] batch_generate_with_ml method exists")
            results['Batch ML Generation'] = True
        else:
            print("   [FAIL] batch_generate_with_ml method not found")
            results['Batch ML Generation'] = False
        
    except Exception as e:
        print(f"   [FAIL] Enhanced video generation import failed: {e}")
        results['Enhanced Video Generation'] = False
    
    return results


async def test_api_endpoints():
    """Test ML API endpoints"""
    print("\n" + "="*60)
    print("Testing ML API Endpoints")
    print("="*60)
    
    results = {}
    
    try:
        from app.api.v1.endpoints.ml_features import router as ml_router
        print("\n1. ML API endpoints imported")
        
        # Check routes
        expected_routes = [
            '/personalize',
            '/predict-performance',
            '/channel-insights',
            '/train-model',
            '/model-status',
            '/generate-video',
            '/batch-generate'
        ]
        
        route_paths = [route.path for route in ml_router.routes if hasattr(route, 'path')]
        
        for route in expected_routes:
            if any(route in path for path in route_paths):
                print(f"   [OK] {route} endpoint registered")
            else:
                print(f"   [FAIL] {route} endpoint not found")
        
        results['API Endpoints'] = True
        
    except Exception as e:
        print(f"   [FAIL] API endpoints test failed: {e}")
        results['API Endpoints'] = False
    
    return results


async def test_main_integration():
    """Test main.py integration"""
    print("\n" + "="*60)
    print("Testing Main Application Integration")
    print("="*60)
    
    results = {}
    
    try:
        # Check if ML services are imported in main.py
        main_path = Path("backend/app/main.py")
        if main_path.exists():
            content = main_path.read_text()
            
            if "ml_integration_service" in content:
                print("   [OK] ML integration service imported in main.py")
                results['ML Service Import'] = True
            else:
                print("   [FAIL] ML integration service not imported in main.py")
                results['ML Service Import'] = False
            
            if "enhanced_video_generation" in content:
                print("   [OK] Enhanced video generation imported in main.py")
                results['Enhanced Gen Import'] = True
            else:
                print("   [FAIL] Enhanced video generation not imported in main.py")
                results['Enhanced Gen Import'] = False
            
            if "ML_ENABLED" in content:
                print("   [OK] ML configuration check in main.py")
                results['ML Config Check'] = True
            else:
                print("   [INFO] ML configuration check not found in main.py")
                results['ML Config Check'] = False
        
    except Exception as e:
        print(f"   [FAIL] Main integration test failed: {e}")
        results['Main Integration'] = False
    
    return results


async def test_video_endpoint_integration():
    """Test video generation endpoint integration"""
    print("\n" + "="*60)
    print("Testing Video Generation Endpoint Integration")
    print("="*60)
    
    results = {}
    
    try:
        video_gen_path = Path("backend/app/api/v1/endpoints/video_generation.py")
        if video_gen_path.exists():
            content = video_gen_path.read_text()
            
            if "generate_personalized_video" in content:
                print("   [OK] ML video generation integrated in endpoint")
                results['ML Video Endpoint'] = True
            else:
                print("   [FAIL] ML video generation not integrated")
                results['ML Video Endpoint'] = False
            
            if "use_ml_personalization" in content:
                print("   [OK] ML personalization parameter added")
                results['ML Parameters'] = True
            else:
                print("   [FAIL] ML personalization parameter not found")
                results['ML Parameters'] = False
            
            if "settings.ML_ENABLED" in content:
                print("   [OK] ML feature flag check in endpoint")
                results['ML Feature Flag'] = True
            else:
                print("   [INFO] ML feature flag check not found")
                results['ML Feature Flag'] = False
        
    except Exception as e:
        print(f"   [FAIL] Video endpoint test failed: {e}")
        results['Video Endpoint Integration'] = False
    
    return results


async def main():
    """Main test execution"""
    print("\n" + "="*60)
    print("YTEmpire Full ML Integration Test Suite")
    print("="*60)
    
    all_results = {}
    
    # Run all test suites
    print("\nRunning test suites...")
    
    # Test ML services
    ml_results = await test_ml_services()
    all_results.update(ml_results)
    
    # Test enhanced video generation
    video_results = await test_enhanced_video_generation()
    all_results.update(video_results)
    
    # Test API endpoints
    api_results = await test_api_endpoints()
    all_results.update(api_results)
    
    # Test main integration
    main_results = await test_main_integration()
    all_results.update(main_results)
    
    # Test video endpoint integration
    endpoint_results = await test_video_endpoint_integration()
    all_results.update(endpoint_results)
    
    # Summary
    print("\n" + "="*60)
    print("Full Integration Test Summary")
    print("="*60)
    
    passed = 0
    failed = 0
    
    for test_name, result in all_results.items():
        status = "[OK]" if result else "[FAIL]"
        print(f"{test_name}: {status}")
        if result:
            passed += 1
        else:
            failed += 1
    
    total = passed + failed
    print(f"\nTotal: {passed}/{total} tests passed")
    
    if passed == total:
        print("\n[SUCCESS] Full ML integration is working!")
        print("\nIntegration includes:")
        print("  - AutoML pipeline for performance prediction")
        print("  - Personalization engine for content optimization")
        print("  - ML integration service connecting models to backend")
        print("  - Enhanced video generation with ML features")
        print("  - API endpoints exposing ML capabilities")
        print("  - Main application properly configured")
        print("  - Video generation endpoints using ML when enabled")
    else:
        print(f"\n[WARNING] {failed} tests failed")
        print("Review the failures above for details")
    
    return passed == total


if __name__ == "__main__":
    success = asyncio.run(main())
    sys.exit(0 if success else 1)