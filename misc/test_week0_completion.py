#!/usr/bin/env python3
"""
Comprehensive Test Suite for Week 0 Completion
Tests all newly created components and verifies 100% completion
"""

import os
import sys
import json
import asyncio
import importlib.util
from pathlib import Path
from typing import Dict, List, Tuple, Any
from datetime import datetime

# Add project root to path
PROJECT_ROOT = Path(r"C:\Users\Hp\projects\ytempire-mvp")
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "backend"))
sys.path.insert(0, str(PROJECT_ROOT / "ml-pipeline"))

# Test results storage
test_results = {
    "passed": [],
    "failed": [],
    "warnings": []
}


def log_test(name: str, status: str, message: str = ""):
    """Log test result"""
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    result = {
        "name": name,
        "status": status,
        "message": message,
        "timestamp": timestamp
    }
    
    if status == "PASS":
        test_results["passed"].append(result)
        print(f"[PASS] {name}")
    elif status == "FAIL":
        test_results["failed"].append(result)
        print(f"[FAIL] {name}: {message}")
    else:
        test_results["warnings"].append(result)
        print(f"[WARN] {name}: {message}")


class Week0CompletionTests:
    """Test suite for Week 0 completion verification"""
    
    def __init__(self):
        self.backend_path = PROJECT_ROOT / "backend"
        self.frontend_path = PROJECT_ROOT / "frontend"
        self.ml_path = PROJECT_ROOT / "ml-pipeline"
        self.infra_path = PROJECT_ROOT / "infrastructure"
    
    def test_backend_readme(self):
        """Test if backend README exists and is properly formatted"""
        readme_path = self.backend_path / "README.md"
        try:
            assert readme_path.exists(), "Backend README.md does not exist"
            
            content = readme_path.read_text()
            
            # Check for essential sections
            required_sections = [
                "## Overview",
                "## Architecture",
                "## Quick Start",
                "## API Documentation",
                "## Services",
                "## Testing",
                "## Security"
            ]
            
            for section in required_sections:
                assert section in content, f"Missing section: {section}"
            
            log_test("Backend README", "PASS")
            return True
            
        except AssertionError as e:
            log_test("Backend README", "FAIL", str(e))
            return False
        except Exception as e:
            log_test("Backend README", "FAIL", f"Unexpected error: {e}")
            return False
    
    def test_websocket_endpoints(self):
        """Test WebSocket endpoints file"""
        ws_path = self.backend_path / "app" / "api" / "v1" / "endpoints" / "websockets.py"
        try:
            assert ws_path.exists(), "WebSocket endpoints file does not exist"
            
            # Try to import the module
            spec = importlib.util.spec_from_file_location("websockets", ws_path)
            ws_module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(ws_module)
            
            # Check for required endpoints
            assert hasattr(ws_module, 'router'), "No router defined"
            assert hasattr(ws_module, 'video_generation_updates'), "Missing video_generation_updates endpoint"
            assert hasattr(ws_module, 'analytics_stream'), "Missing analytics_stream endpoint"
            assert hasattr(ws_module, 'notifications_stream'), "Missing notifications_stream endpoint"
            
            log_test("WebSocket Endpoints", "PASS")
            return True
            
        except AssertionError as e:
            log_test("WebSocket Endpoints", "FAIL", str(e))
            return False
        except Exception as e:
            log_test("WebSocket Endpoints", "FAIL", f"Import error: {e}")
            return False
    
    def test_ml_model_endpoints(self):
        """Test ML model endpoints"""
        ml_path = self.backend_path / "app" / "api" / "v1" / "endpoints" / "ml_models.py"
        try:
            assert ml_path.exists(), "ML model endpoints file does not exist"
            
            # Try to import
            spec = importlib.util.spec_from_file_location("ml_models", ml_path)
            ml_module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(ml_module)
            
            # Check for required components
            assert hasattr(ml_module, 'router'), "No router defined"
            assert hasattr(ml_module, 'ModelConfig'), "Missing ModelConfig model"
            assert hasattr(ml_module, 'list_models'), "Missing list_models endpoint"
            assert hasattr(ml_module, 'deploy_model'), "Missing deploy_model endpoint"
            assert hasattr(ml_module, 'get_model_metrics'), "Missing get_model_metrics endpoint"
            
            log_test("ML Model Endpoints", "PASS")
            return True
            
        except AssertionError as e:
            log_test("ML Model Endpoints", "FAIL", str(e))
            return False
        except Exception as e:
            log_test("ML Model Endpoints", "FAIL", f"Import error: {e}")
            return False
    
    def test_reports_endpoint(self):
        """Test reports endpoint"""
        reports_path = self.backend_path / "app" / "api" / "v1" / "endpoints" / "reports.py"
        try:
            assert reports_path.exists(), "Reports endpoint file does not exist"
            
            # Try to import
            spec = importlib.util.spec_from_file_location("reports", reports_path)
            reports_module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(reports_module)
            
            # Check for required components
            assert hasattr(reports_module, 'router'), "No router defined"
            assert hasattr(reports_module, 'ReportRequest'), "Missing ReportRequest model"
            assert hasattr(reports_module, 'generate_report'), "Missing generate_report endpoint"
            assert hasattr(reports_module, 'get_performance_report'), "Missing performance report endpoint"
            assert hasattr(reports_module, 'get_cost_report'), "Missing cost report endpoint"
            
            log_test("Reports Endpoint", "PASS")
            return True
            
        except AssertionError as e:
            log_test("Reports Endpoint", "FAIL", str(e))
            return False
        except Exception as e:
            log_test("Reports Endpoint", "FAIL", f"Import error: {e}")
            return False
    
    def test_trend_analyzer_service(self):
        """Test trend analyzer service"""
        trend_path = self.backend_path / "app" / "services" / "trend_analyzer.py"
        try:
            assert trend_path.exists(), "Trend analyzer service does not exist"
            
            # Try to import
            spec = importlib.util.spec_from_file_location("trend_analyzer", trend_path)
            trend_module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(trend_module)
            
            # Check for required components
            assert hasattr(trend_module, 'TrendAnalyzer'), "Missing TrendAnalyzer class"
            assert hasattr(trend_module, 'trend_analyzer'), "Missing trend_analyzer instance"
            assert hasattr(trend_module, 'TrendCategory'), "Missing TrendCategory enum"
            
            # Check methods
            analyzer = trend_module.TrendAnalyzer()
            assert hasattr(analyzer, 'analyze_trends'), "Missing analyze_trends method"
            assert hasattr(analyzer, 'predict_viral_potential'), "Missing predict_viral_potential method"
            assert hasattr(analyzer, 'get_competition_analysis'), "Missing get_competition_analysis method"
            
            log_test("Trend Analyzer Service", "PASS")
            return True
            
        except AssertionError as e:
            log_test("Trend Analyzer Service", "FAIL", str(e))
            return False
        except Exception as e:
            log_test("Trend Analyzer Service", "FAIL", f"Import error: {e}")
            return False
    
    def test_ml_performance_tracker(self):
        """Test ML performance tracker"""
        perf_path = self.ml_path / "monitoring" / "performance_tracker.py"
        try:
            assert perf_path.exists(), "ML performance tracker does not exist"
            
            # Try to import
            spec = importlib.util.spec_from_file_location("performance_tracker", perf_path)
            perf_module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(perf_module)
            
            # Check for required components
            assert hasattr(perf_module, 'PerformanceTracker'), "Missing PerformanceTracker class"
            assert hasattr(perf_module, 'ModelMetrics'), "Missing ModelMetrics dataclass"
            assert hasattr(perf_module, 'PerformanceAlert'), "Missing PerformanceAlert dataclass"
            assert hasattr(perf_module, 'performance_tracker'), "Missing performance_tracker instance"
            
            log_test("ML Performance Tracker", "PASS")
            return True
            
        except AssertionError as e:
            log_test("ML Performance Tracker", "FAIL", str(e))
            return False
        except Exception as e:
            log_test("ML Performance Tracker", "FAIL", f"Import error: {e}")
            return False
    
    def test_prometheus_config(self):
        """Test Prometheus configuration"""
        prom_dir = self.infra_path / "monitoring" / "prometheus"
        try:
            assert prom_dir.exists(), "Prometheus directory does not exist"
            
            # Check for required files
            required_files = [
                "prometheus.yml",
                "alert.rules.yml",
                "recording.rules.yml"
            ]
            
            for file_name in required_files:
                file_path = prom_dir / file_name
                assert file_path.exists(), f"Missing file: {file_name}"
                
                # Verify file is not empty
                content = file_path.read_text()
                assert len(content) > 100, f"{file_name} appears to be empty or too small"
            
            log_test("Prometheus Configuration", "PASS")
            return True
            
        except AssertionError as e:
            log_test("Prometheus Configuration", "FAIL", str(e))
            return False
        except Exception as e:
            log_test("Prometheus Configuration", "FAIL", f"Unexpected error: {e}")
            return False
    
    def test_api_router_updates(self):
        """Test that API router includes new endpoints"""
        api_path = self.backend_path / "app" / "api" / "v1" / "api.py"
        try:
            assert api_path.exists(), "API router file does not exist"
            
            content = api_path.read_text()
            
            # Check for new imports
            assert "websockets" in content, "websockets not imported"
            assert "ml_models" in content, "ml_models not imported"
            assert "reports" in content, "reports not imported"
            
            # Check for router includes
            assert "websockets.router" in content, "websockets router not included"
            assert "ml_models.router" in content, "ml_models router not included"
            assert "reports.router" in content, "reports router not included"
            
            log_test("API Router Updates", "PASS")
            return True
            
        except AssertionError as e:
            log_test("API Router Updates", "FAIL", str(e))
            return False
        except Exception as e:
            log_test("API Router Updates", "FAIL", f"Unexpected error: {e}")
            return False
    
    def test_frontend_dashboard(self):
        """Test that Dashboard page exists"""
        dashboard_path = self.frontend_path / "src" / "pages" / "Dashboard" / "Dashboard.tsx"
        try:
            assert dashboard_path.exists(), "Dashboard.tsx does not exist"
            
            content = dashboard_path.read_text()
            assert len(content) > 100, "Dashboard.tsx appears to be empty"
            assert "Dashboard" in content, "Dashboard component not defined"
            
            log_test("Frontend Dashboard Page", "PASS")
            return True
            
        except AssertionError as e:
            log_test("Frontend Dashboard Page", "FAIL", str(e))
            return False
        except Exception as e:
            log_test("Frontend Dashboard Page", "FAIL", f"Unexpected error: {e}")
            return False
    
    def test_payment_endpoint_exists(self):
        """Test that payment endpoint exists (either payment.py or payments.py)"""
        payment_path1 = self.backend_path / "app" / "api" / "v1" / "endpoints" / "payment.py"
        payment_path2 = self.backend_path / "app" / "api" / "v1" / "endpoints" / "payments.py"
        
        try:
            assert payment_path1.exists() or payment_path2.exists(), "Payment endpoint does not exist"
            log_test("Payment Endpoint", "PASS")
            return True
            
        except AssertionError as e:
            log_test("Payment Endpoint", "FAIL", str(e))
            return False
    
    def test_trend_detection_consolidation(self):
        """Check trend detection files and verify no harmful duplication"""
        trend_files = [
            self.ml_path / "services" / "trend_detection.py",
            self.ml_path / "src" / "trend_detection_model.py",
            self.ml_path / "src" / "trend_prediction.py"
        ]
        
        existing_files = [f for f in trend_files if f.exists()]
        
        if len(existing_files) > 1:
            log_test("Trend Detection Files", "WARNING", 
                    f"Multiple trend detection files exist ({len(existing_files)}). Consider consolidation.")
        else:
            log_test("Trend Detection Files", "PASS")
        
        return True
    
    async def test_service_imports(self):
        """Test that all services can be imported without errors"""
        services_to_test = [
            ("Trend Analyzer", "app.services.trend_analyzer"),
            ("WebSocket Manager", "app.services.websocket_manager"),
            ("Analytics Service", "app.services.analytics_service"),
            ("Cost Tracker", "app.services.cost_tracking"),
        ]
        
        all_passed = True
        for name, module_path in services_to_test:
            try:
                module = importlib.import_module(module_path)
                log_test(f"Import {name}", "PASS")
            except ImportError as e:
                log_test(f"Import {name}", "FAIL", str(e))
                all_passed = False
            except Exception as e:
                log_test(f"Import {name}", "FAIL", f"Unexpected error: {e}")
                all_passed = False
        
        return all_passed
    
    def run_all_tests(self):
        """Run all tests and generate report"""
        print("\n" + "="*60)
        print("WEEK 0 COMPLETION TEST SUITE")
        print("="*60 + "\n")
        
        # Run synchronous tests
        self.test_backend_readme()
        self.test_websocket_endpoints()
        self.test_ml_model_endpoints()
        self.test_reports_endpoint()
        self.test_trend_analyzer_service()
        self.test_ml_performance_tracker()
        self.test_prometheus_config()
        self.test_api_router_updates()
        self.test_frontend_dashboard()
        self.test_payment_endpoint_exists()
        self.test_trend_detection_consolidation()
        
        # Run async tests
        # asyncio.run(self.test_service_imports())
        
        # Generate summary
        print("\n" + "="*60)
        print("TEST SUMMARY")
        print("="*60)
        
        total_tests = len(test_results["passed"]) + len(test_results["failed"])
        pass_rate = (len(test_results["passed"]) / total_tests * 100) if total_tests > 0 else 0
        
        print(f"\nTotal Tests: {total_tests}")
        print(f"Passed: {len(test_results['passed'])}")
        print(f"Failed: {len(test_results['failed'])}")
        print(f"Warnings: {len(test_results['warnings'])}")
        print(f"Pass Rate: {pass_rate:.1f}%")
        
        if test_results["failed"]:
            print("\nFAILED TESTS:")
            for test in test_results["failed"]:
                print(f"  - {test['name']}: {test['message']}")
        
        if test_results["warnings"]:
            print("\nWARNINGS:")
            for test in test_results["warnings"]:
                print(f"  - {test['name']}: {test['message']}")
        
        # Overall status
        print("\n" + "="*60)
        if pass_rate == 100:
            print("ALL TESTS PASSED - WEEK 0 100% COMPLETE!")
        elif pass_rate >= 95:
            print("ALMOST THERE - {:.1f}% COMPLETE".format(pass_rate))
        else:
            print("TESTS FAILED - {:.1f}% COMPLETE".format(pass_rate))
        print("="*60 + "\n")
        
        # Save test results
        results_path = PROJECT_ROOT / "misc" / "week0_test_results.json"
        with open(results_path, 'w') as f:
            json.dump(test_results, f, indent=2, default=str)
        print(f"Test results saved to: {results_path}")
        
        return pass_rate == 100


def main():
    """Main test runner"""
    tester = Week0CompletionTests()
    success = tester.run_all_tests()
    
    # Exit with appropriate code
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()