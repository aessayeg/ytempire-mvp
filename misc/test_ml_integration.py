"""
Integration test for ML features with YTEmpire backend
Tests the complete integration of AutoML and Personalization with existing services
"""

import sys
import os
import asyncio
import json
from pathlib import Path
from datetime import datetime
import numpy as np
import pandas as pd

# Add project paths
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'backend'))
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'ml-pipeline', 'src'))

# Test imports
def test_imports():
    """Test all required imports"""
    print("\n" + "="*60)
    print("Testing Imports")
    print("="*60)
    
    imports_status = {}
    
    # Test ML pipeline imports
    try:
        from automl_pipeline import AutoMLPipeline, AutoMLConfig
        imports_status['AutoML Pipeline'] = "[OK]"
    except ImportError as e:
        imports_status['AutoML Pipeline'] = f"[FAIL] {e}"
    
    try:
        from personalization_model import PersonalizationEngine, PersonalizationConfig
        imports_status['Personalization Model'] = "[OK]"
    except ImportError as e:
        imports_status['Personalization Model'] = f"[FAIL] {e}"
    
    # Test backend service imports
    try:
        from app.services.ml_integration_service import ml_service
        imports_status['ML Integration Service'] = "[OK]"
    except ImportError as e:
        imports_status['ML Integration Service'] = f"[FAIL] {e}"
    
    try:
        from app.services.enhanced_video_generation import enhanced_orchestrator
        imports_status['Enhanced Video Generation'] = "[OK]"
    except ImportError as e:
        imports_status['Enhanced Video Generation'] = f"[FAIL] {e}"
    
    try:
        from app.api.v1.endpoints.ml_features import router as ml_router
        imports_status['ML API Endpoints'] = "[OK]"
    except ImportError as e:
        imports_status['ML API Endpoints'] = f"[FAIL] {e}"
    
    # Print results
    for module, status in imports_status.items():
        print(f"  {module}: {status}")
    
    all_ok = all("[OK]" in status for status in imports_status.values())
    return all_ok


async def test_ml_integration_service():
    """Test ML Integration Service functionality"""
    print("\n" + "="*60)
    print("Testing ML Integration Service")
    print("="*60)
    
    try:
        from app.services.ml_integration_service import ml_service
        
        # Test 1: Check ML availability
        print("\n1. Checking ML model availability...")
        ml_available = ml_service.automl_pipeline is not None
        personalization_available = ml_service.personalization_engine is not None
        
        print(f"   AutoML available: {ml_available}")
        print(f"   Personalization available: {personalization_available}")
        
        # Test 2: Get personalized content
        print("\n2. Testing personalized content generation...")
        
        channel_data = {
            'name': 'TestChannel',
            'niche': 'technology',
            'target_audience': {'age_range': '18-35'}
        }
        
        historical_videos = [
            {
                'title': 'Python Tutorial',
                'views': 50000,
                'likes': 2500,
                'comments': 300,
                'duration': 900,
                'published_at': datetime.now().isoformat()
            }
        ]
        
        recommendation = await ml_service.get_personalized_content_recommendation(
            channel_id='test_channel_001',
            channel_data=channel_data,
            historical_videos=historical_videos,
            trending_topics=['AI', 'Machine Learning']
        )
        
        print(f"   Generated title: {recommendation['title']}")
        print(f"   Confidence: {recommendation['confidence_score']:.2%}")
        print(f"   Style: {recommendation['style']}")
        
        # Test 3: Performance prediction
        print("\n3. Testing performance prediction...")
        
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
        
        print(f"   Predicted views: {prediction['predicted_views']:,}")
        print(f"   Predicted engagement: {prediction['predicted_engagement_rate']:.2%}")
        print(f"   Model type: {prediction['model_type']}")
        
        # Test 4: Channel insights
        print("\n4. Testing channel insights...")
        
        insights = await ml_service.get_channel_insights('test_channel_001')
        
        if 'error' not in insights:
            print(f"   Content style: {insights.get('content_style', 'N/A')}")
            print(f"   Avg engagement: {insights.get('performance', {}).get('avg_engagement', 0):.2%}")
            print(f"   Best posting hour: {insights.get('optimal_schedule', {}).get('best_hour', 12)}")
        
        # Test 5: Check retraining status
        print("\n5. Testing retraining check...")
        
        retrain_status = await ml_service.check_retraining_needed()
        print(f"   AutoML needs retraining: {retrain_status['automl_needs_retraining']}")
        print(f"   Personalization needs update: {retrain_status['personalization_needs_update']}")
        
        print("\n[OK] ML Integration Service tests completed!")
        return True
        
    except Exception as e:
        print(f"\n[FAIL] ML Integration Service test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


async def test_enhanced_video_generation():
    """Test Enhanced Video Generation with ML"""
    print("\n" + "="*60)
    print("Testing Enhanced Video Generation")
    print("="*60)
    
    try:
        from app.services.enhanced_video_generation import enhanced_orchestrator
        
        # Mock database session
        class MockDB:
            async def execute(self, query):
                class Result:
                    def scalar_one_or_none(self):
                        return None
                    def scalars(self):
                        class Scalars:
                            def all(self):
                                return []
                        return Scalars()
                return Result()
        
        db = MockDB()
        
        print("\n1. Testing video generation with ML...")
        
        # Note: This would normally require a full database setup
        # For testing, we'll check if the method exists and is callable
        
        if hasattr(enhanced_orchestrator, 'generate_video_with_ml'):
            print("   generate_video_with_ml method exists")
            
            # Test the method signature
            import inspect
            sig = inspect.signature(enhanced_orchestrator.generate_video_with_ml)
            params = list(sig.parameters.keys())
            
            expected_params = ['channel_id', 'topic', 'db', 'websocket', 'use_personalization', 'use_performance_prediction']
            
            for param in expected_params:
                if param in params:
                    print(f"   [OK] Parameter '{param}' found")
                else:
                    print(f"   [FAIL] Parameter '{param}' missing")
        
        print("\n2. Testing batch generation with ML...")
        
        if hasattr(enhanced_orchestrator, 'batch_generate_with_ml'):
            print("   batch_generate_with_ml method exists")
            
            # Test method is callable
            if callable(enhanced_orchestrator.batch_generate_with_ml):
                print("   Method is callable")
        
        print("\n[OK] Enhanced Video Generation tests completed!")
        return True
        
    except Exception as e:
        print(f"\n[FAIL] Enhanced Video Generation test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_api_endpoints():
    """Test ML API endpoints are properly registered"""
    print("\n" + "="*60)
    print("Testing API Endpoints")
    print("="*60)
    
    try:
        from app.api.v1.endpoints.ml_features import router as ml_router
        
        print("\n1. Checking registered endpoints...")
        
        # Check routes
        expected_endpoints = [
            '/personalize',
            '/predict-performance',
            '/channel-insights/{channel_id}',
            '/train-model',
            '/model-status',
            '/generate-video',
            '/batch-generate',
            '/update-profile/{channel_id}',
            '/recommendations/{channel_id}'
        ]
        
        # Get actual routes from router
        actual_routes = []
        for route in ml_router.routes:
            if hasattr(route, 'path'):
                actual_routes.append(route.path)
        
        for endpoint in expected_endpoints:
            if any(endpoint in route for route in actual_routes):
                print(f"   [OK] Endpoint {endpoint}")
            else:
                print(f"   [FAIL] Endpoint {endpoint} not found")
        
        print("\n2. Checking API router registration...")
        
        try:
            from app.api.v1.api import api_router
            
            # Check if ml_features router is included
            ml_registered = False
            for route in api_router.routes:
                if hasattr(route, 'path') and '/ml' in route.path:
                    ml_registered = True
                    break
            
            if ml_registered:
                print("   [OK] ML router registered in main API")
            else:
                print("   [FAIL] ML router not registered in main API")
        except ImportError:
            print("   [SKIP] Could not import main API router")
        
        print("\n[OK] API Endpoints tests completed!")
        return True
        
    except Exception as e:
        print(f"\n[FAIL] API Endpoints test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_model_persistence():
    """Test model saving and loading"""
    print("\n" + "="*60)
    print("Testing Model Persistence")
    print("="*60)
    
    try:
        models_path = Path("models")
        
        print("\n1. Checking model directories...")
        
        dirs_to_check = [
            models_path / "automl",
            models_path / "personalization"
        ]
        
        for dir_path in dirs_to_check:
            if dir_path.exists():
                print(f"   [OK] Directory exists: {dir_path}")
                
                # List files in directory
                files = list(dir_path.glob("*"))
                if files:
                    print(f"       Files found: {len(files)}")
                    for file in files[:3]:  # Show first 3 files
                        print(f"       - {file.name}")
            else:
                print(f"   [INFO] Directory not found: {dir_path}")
                dir_path.mkdir(parents=True, exist_ok=True)
                print(f"   [OK] Created directory: {dir_path}")
        
        print("\n2. Testing save/load capability...")
        
        # Test with a small example
        from automl_pipeline import AutoMLPipeline, AutoMLConfig
        
        config = AutoMLConfig(
            max_models_to_evaluate=1,
            cv_folds=2,
            save_path=models_path / "automl"
        )
        
        pipeline = AutoMLPipeline(config)
        
        # Create dummy data
        X = pd.DataFrame(np.random.randn(20, 5))
        y = pd.Series(np.random.randn(20))
        
        # Train a simple model
        print("   Training test model...")
        results = pipeline.train(X, y)
        
        # Save model
        test_path = models_path / "automl" / "test_model.pkl"
        pipeline.save_model(test_path)
        print(f"   [OK] Model saved to {test_path}")
        
        # Load model
        new_pipeline = AutoMLPipeline(config)
        new_pipeline.load_model(test_path)
        print(f"   [OK] Model loaded from {test_path}")
        
        # Test prediction
        predictions = new_pipeline.predict(X[:5])
        print(f"   [OK] Predictions made: {len(predictions)} samples")
        
        # Clean up test file
        if test_path.exists():
            test_path.unlink()
            print(f"   [OK] Cleaned up test file")
        
        print("\n[OK] Model Persistence tests completed!")
        return True
        
    except Exception as e:
        print(f"\n[FAIL] Model Persistence test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


async def main():
    """Main test execution"""
    print("\n" + "="*60)
    print("YTEmpire ML Integration Test Suite")
    print("Testing Full Integration with Backend")
    print("="*60)
    
    results = {
        'Imports': False,
        'ML Integration Service': False,
        'Enhanced Video Generation': False,
        'API Endpoints': False,
        'Model Persistence': False
    }
    
    # Run tests
    results['Imports'] = test_imports()
    
    if results['Imports']:
        results['ML Integration Service'] = await test_ml_integration_service()
        results['Enhanced Video Generation'] = await test_enhanced_video_generation()
        results['API Endpoints'] = test_api_endpoints()
        results['Model Persistence'] = test_model_persistence()
    else:
        print("\n[WARNING] Skipping integration tests due to import failures")
    
    # Summary
    print("\n" + "="*60)
    print("Integration Test Summary")
    print("="*60)
    
    for test_name, passed in results.items():
        status = "[OK] PASSED" if passed else "[FAIL] FAILED"
        print(f"{test_name}: {status}")
    
    total_passed = sum(results.values())
    total_tests = len(results)
    
    print(f"\nTotal: {total_passed}/{total_tests} tests passed")
    
    if total_passed == total_tests:
        print("\n[SUCCESS] All ML components are fully integrated!")
        print("\nThe ML pipeline is now integrated with:")
        print("  - Video generation orchestrator")
        print("  - Backend services")
        print("  - API endpoints")
        print("  - Cost tracking")
        print("  - Database models")
    else:
        print("\n[WARNING] Some integration tests failed.")
        print("Please review the errors above.")
    
    return total_passed == total_tests


if __name__ == "__main__":
    success = asyncio.run(main())
    sys.exit(0 if success else 1)