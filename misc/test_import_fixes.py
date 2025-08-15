"""
Test all import fixes to ensure they work correctly
"""

import sys
import os
from pathlib import Path

# Add backend to path
backend_path = Path(__file__).parent.parent / "backend"
sys.path.insert(0, str(backend_path))

def test_imports():
    """Test all critical imports that were fixed"""
    
    results = {
        "success": [],
        "failed": []
    }
    
    print("\n" + "="*80)
    print("TESTING IMPORT FIXES")
    print("="*80)
    
    # Test cases for all fixed imports
    test_cases = [
        # Basic service imports
        ("app.services.analytics_service", "analytics_service", "AnalyticsService"),
        ("app.services.cost_tracking", "cost_tracker", "CostTracker"),
        ("app.services.video_generation_pipeline", "VideoGenerationPipeline", None),
        ("app.services.video_generation_orchestrator", "video_orchestrator", None),
        ("app.services.enhanced_video_generation", "enhanced_orchestrator", None),
        ("app.services.video_processor", "video_processor", None),
        
        # Aliased imports
        ("app.services.realtime_analytics_service", "realtime_analytics_service", None),
        ("app.services.training_data_service", "training_data_service", None),
        
        # Core services
        ("app.core.config", "settings", None),
        ("app.core.database", None, None),
        ("app.db.session", "engine", None),
        
        # WebSocket
        ("app.services.websocket_manager", "ConnectionManager", None),
        
        # Other critical services
        ("app.services.youtube_multi_account", "get_youtube_manager", None),
        ("app.services.batch_processing", "batch_processor", None),
        ("app.services.notification_service", "notification_service", None),
    ]
    
    for module_name, attribute, class_name in test_cases:
        try:
            # Try to import the module
            module = __import__(module_name, fromlist=[attribute or class_name or ''])
            
            # Check if specific attribute exists
            if attribute:
                if hasattr(module, attribute):
                    results["success"].append(f"{module_name}.{attribute}")
                    print(f"  [OK] {module_name}.{attribute}")
                else:
                    results["failed"].append(f"{module_name}.{attribute} - attribute not found")
                    print(f"  [FAIL] {module_name}.{attribute} - attribute not found")
            elif class_name:
                if hasattr(module, class_name):
                    results["success"].append(f"{module_name}.{class_name}")
                    print(f"  [OK] {module_name}.{class_name}")
                else:
                    results["failed"].append(f"{module_name}.{class_name} - class not found")
                    print(f"  [FAIL] {module_name}.{class_name} - class not found")
            else:
                results["success"].append(module_name)
                print(f"  [OK] {module_name}")
                
        except ImportError as e:
            results["failed"].append(f"{module_name} - {str(e)}")
            print(f"  [FAIL] {module_name} - Import Error: {str(e)}")
        except Exception as e:
            results["failed"].append(f"{module_name} - {str(e)}")
            print(f"  [FAIL] {module_name} - Error: {str(e)}")
    
    # Test specific aliased imports from main.py
    print("\n" + "-"*80)
    print("TESTING MAIN.PY SPECIFIC ALIASES")
    print("-"*80)
    
    alias_tests = [
        ("from app.services.analytics_service import analytics_service as quality_monitor", "quality_monitor"),
        ("from app.services.cost_tracking import cost_tracker as revenue_tracking_service", "revenue_tracking_service"),
        ("from app.services.analytics_service import analytics_service as analytics_connector", "analytics_connector"),
        ("from app.services.cost_tracking import cost_tracker as cost_aggregator", "cost_aggregator"),
        ("from app.services.analytics_service import analytics_service as metrics_aggregator", "metrics_aggregator"),
        ("from app.services.cost_tracking import cost_tracker as cost_verifier", "cost_verifier"),
    ]
    
    for import_statement, alias_name in alias_tests:
        try:
            exec(import_statement)
            results["success"].append(f"Alias: {alias_name}")
            print(f"  [OK] {alias_name} alias works")
        except Exception as e:
            results["failed"].append(f"Alias {alias_name}: {str(e)}")
            print(f"  [FAIL] {alias_name} alias failed: {str(e)}")
    
    # Summary
    print("\n" + "="*80)
    print("IMPORT TEST SUMMARY")
    print("="*80)
    
    total_tests = len(results["success"]) + len(results["failed"])
    success_rate = (len(results["success"]) / total_tests * 100) if total_tests > 0 else 0
    
    print(f"\nTotal Tests: {total_tests}")
    print(f"Successful: {len(results['success'])}")
    print(f"Failed: {len(results['failed'])}")
    print(f"Success Rate: {success_rate:.1f}%")
    
    if results["failed"]:
        print("\nFailed Imports:")
        for failure in results["failed"][:10]:  # Show first 10 failures
            print(f"  - {failure}")
    
    return results


if __name__ == "__main__":
    results = test_imports()
    
    # Exit with appropriate code
    if results["failed"]:
        sys.exit(1)
    else:
        print("\n[SUCCESS] ALL IMPORTS WORKING CORRECTLY!")
        sys.exit(0)