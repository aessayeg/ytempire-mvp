#!/usr/bin/env python3
"""
Test script to verify all services can be imported and the backend can start
"""

import sys
import importlib
import traceback
from pathlib import Path

# Add backend to path
backend_path = Path(__file__).parent.parent / "backend"
sys.path.insert(0, str(backend_path))

def test_service_imports():
    """Test that all services can be imported without errors"""
    print("=== TESTING SERVICE IMPORTS ===\n")
    
    services = [
        # Critical infrastructure services
        "app.services.cost_tracking",
        "app.services.gpu_resource_service", 
        "app.services.youtube_multi_account",
        "app.services.alert_service",
        
        # Core business services
        "app.services.analytics_service",
        "app.services.revenue_tracking",
        "app.services.quality_metrics",
        "app.services.video_generation_orchestrator",
        
        # Services with async initialize
        "app.services.analytics_connector",
        "app.services.analytics_pipeline",
        "app.services.cost_aggregation",
        "app.services.data_lake_service",
        "app.services.export_service",
        "app.services.feature_store",
        "app.services.gpu_resource_manager",
        "app.services.inference_pipeline",
        "app.services.training_data_service",
        
        # Utility services
        "app.services.notification_service",
        "app.services.api_optimization",
        "app.services.batch_processing",
        "app.services.storage_service",
        "app.services.thumbnail_generator",
        "app.services.stock_footage",
        "app.services.quick_video_generator",
        "app.services.rate_limiter",
        "app.services.websocket_manager",
        "app.services.user_behavior_analytics",
        "app.services.automated_reporting",
        "app.services.defect_tracking",
        "app.services.model_monitoring",
        "app.services.vector_database",
        "app.services.roi_calculator",
        "app.services.performance_monitoring",
        
        # Additional services
        "app.services.advanced_caching",
        "app.services.error_handlers",
        "app.services.feature_engineering",
        "app.services.metrics_aggregation",
        "app.services.metrics_pipeline",
        "app.services.n8n_integration",
        "app.services.optimized_queries",
        "app.services.prompt_engineering",
        "app.services.reporting",
        "app.services.video_processor",
        "app.services.webhook_service",
        "app.services.websocket_events",
        "app.services.ab_testing_service",
        "app.services.cost_optimizer",
        "app.services.cost_verification",
        "app.services.mock_video_generator",
        "app.services.video_generation_pipeline"
    ]
    
    success_count = 0
    failed_imports = []
    
    for service in services:
        try:
            importlib.import_module(service)
            print(f"PASS {service}")
            success_count += 1
        except Exception as e:
            print(f"FAIL {service}: {str(e)}")
            failed_imports.append((service, str(e)))
    
    print(f"\n=== IMPORT RESULTS ===")
    print(f"Successful: {success_count}/{len(services)} ({success_count/len(services)*100:.1f}%)")
    print(f"Failed: {len(failed_imports)}/{len(services)} ({len(failed_imports)/len(services)*100:.1f}%)")
    
    if failed_imports:
        print(f"\n=== FAILED IMPORTS ===")
        for service, error in failed_imports:
            print(f"FAIL {service}: {error}")
    
    return success_count == len(services)

def test_main_import():
    """Test that main.py can be imported (which tests all service integrations)"""
    print(f"\n=== TESTING MAIN.PY IMPORT ===\n")
    
    try:
        import app.main
        print("PASS app.main imported successfully")
        print("PASS All service integrations working")
        return True
    except Exception as e:
        print(f"FAIL Failed to import app.main: {str(e)}")
        print("\n=== FULL TRACEBACK ===")
        traceback.print_exc()
        return False

def main():
    print("TESTING ALL SERVICE INTEGRATIONS\n")
    
    # Test individual service imports
    imports_ok = test_service_imports()
    
    # Test main.py import (integration test)
    main_ok = test_main_import()
    
    print(f"\n=== FINAL RESULTS ===")
    print(f"Service imports: {'PASS' if imports_ok else 'FAIL'}")
    print(f"Main integration: {'PASS' if main_ok else 'FAIL'}")
    
    if imports_ok and main_ok:
        print(f"\nALL SERVICES SUCCESSFULLY INTEGRATED!")
        print(f"Backend should start without errors.")
    else:
        print(f"\nINTEGRATION ISSUES DETECTED!")
        print(f"Fix the above errors before proceeding.")
    
    return imports_ok and main_ok

if __name__ == '__main__':
    success = main()
    sys.exit(0 if success else 1)