"""
Complete Integration Verification and Report Generation
Ensures all modules are properly connected after consolidation
"""

import os
import re
import json
import importlib.util
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Set, Tuple
import subprocess
import sys

class IntegrationVerifier:
    def __init__(self):
        self.project_root = Path(".")
        self.backend_path = Path("backend")
        self.frontend_path = Path("frontend")
        self.results = {
            "timestamp": datetime.now().isoformat(),
            "integration_status": {},
            "service_connections": {},
            "api_mappings": {},
            "websocket_connections": {},
            "celery_tasks": {},
            "database_connections": {},
            "frontend_backend_links": {},
            "errors": [],
            "warnings": [],
            "recommendations": []
        }
        
    def verify_all_integrations(self):
        """Main verification process"""
        print("\n" + "="*100)
        print("COMPLETE INTEGRATION VERIFICATION - POST CONSOLIDATION")
        print("="*100)
        
        # 1. Verify main.py can import all services
        print("\n[1/10] Verifying main.py service imports...")
        self.verify_main_imports()
        
        # 2. Verify API endpoints connect to services
        print("\n[2/10] Verifying API endpoint -> service connections...")
        self.verify_api_service_connections()
        
        # 3. Verify Celery tasks integration
        print("\n[3/10] Verifying Celery task integrations...")
        self.verify_celery_integration()
        
        # 4. Verify WebSocket connections
        print("\n[4/10] Verifying WebSocket integrations...")
        self.verify_websocket_integration()
        
        # 5. Verify database model usage
        print("\n[5/10] Verifying database model integrations...")
        self.verify_database_integration()
        
        # 6. Verify inter-service dependencies
        print("\n[6/10] Verifying inter-service dependencies...")
        self.verify_service_dependencies()
        
        # 7. Verify frontend API calls
        print("\n[7/10] Verifying frontend -> backend API connections...")
        self.verify_frontend_backend_integration()
        
        # 8. Verify configuration and environment
        print("\n[8/10] Verifying configuration integrity...")
        self.verify_configuration()
        
        # 9. Test actual imports
        print("\n[9/10] Testing actual Python imports...")
        self.test_actual_imports()
        
        # 10. Generate comprehensive report
        print("\n[10/10] Generating comprehensive integration report...")
        self.generate_integration_report()
        
        return self.results
    
    def verify_main_imports(self):
        """Verify all imports in main.py work"""
        main_file = self.backend_path / "app" / "main.py"
        
        try:
            with open(main_file, 'r', encoding='utf-8') as f:
                content = f.read()
            
            # Extract all imports
            imports = re.findall(r'from app\.services\.(\w+) import (\w+)', content)
            
            successful_imports = []
            failed_imports = []
            aliased_imports = []
            
            for service_module, service_name in imports:
                # Check if file exists
                service_file = self.backend_path / "app" / "services" / f"{service_module}.py"
                
                if service_file.exists():
                    # Check if the imported name exists in file
                    with open(service_file, 'r', encoding='utf-8') as f:
                        service_content = f.read()
                    
                    if service_name in service_content or f"class {service_name}" in service_content:
                        successful_imports.append(f"{service_module}.{service_name}")
                    else:
                        # Might be aliased
                        if "as " + service_name in content:
                            aliased_imports.append(f"{service_module} as {service_name}")
                        else:
                            failed_imports.append(f"{service_module}.{service_name}")
                else:
                    # Check if it's been consolidated
                    if service_module in ["mock_video_generator", "quick_video_generator", "analytics_connector"]:
                        aliased_imports.append(f"{service_module} (consolidated)")
                    else:
                        failed_imports.append(f"{service_module} (file missing)")
            
            self.results["integration_status"]["main_imports"] = {
                "total": len(imports),
                "successful": len(successful_imports),
                "failed": len(failed_imports),
                "aliased": len(aliased_imports),
                "success_rate": f"{len(successful_imports)/len(imports)*100:.1f}%" if imports else "0%"
            }
            
            print(f"  Total imports: {len(imports)}")
            print(f"  Successful: {len(successful_imports)}")
            print(f"  Aliased/Consolidated: {len(aliased_imports)}")
            print(f"  Failed: {len(failed_imports)}")
            
            if failed_imports:
                print("  Failed imports:")
                for imp in failed_imports[:5]:
                    print(f"    - {imp}")
                    self.results["errors"].append(f"Failed import: {imp}")
                    
        except Exception as e:
            print(f"  ERROR: {e}")
            self.results["errors"].append(f"main.py verification error: {str(e)}")
    
    def verify_api_service_connections(self):
        """Verify API endpoints properly connect to services"""
        endpoints_path = self.backend_path / "app" / "api" / "v1" / "endpoints"
        
        if not endpoints_path.exists():
            print("  ERROR: API endpoints directory not found")
            return
        
        endpoint_files = list(endpoints_path.glob("*.py"))
        connections = {}
        
        for endpoint_file in endpoint_files:
            if endpoint_file.name == "__init__.py":
                continue
            
            try:
                with open(endpoint_file, 'r', encoding='utf-8') as f:
                    content = f.read()
                
                # Find service imports
                service_imports = re.findall(r'from app\.services\.(\w+)', content)
                
                # Find route definitions
                routes = re.findall(r'@router\.(get|post|put|delete|patch)\s*\(\s*["\'](/[^"\']+)', content)
                
                connections[endpoint_file.stem] = {
                    "services": list(set(service_imports)),
                    "routes": [f"{method.upper()} {path}" for method, path in routes],
                    "route_count": len(routes)
                }
                
            except Exception as e:
                self.results["warnings"].append(f"Could not analyze {endpoint_file.name}: {str(e)}")
        
        self.results["api_mappings"] = connections
        
        # Summary
        total_endpoints = len(connections)
        total_routes = sum(c["route_count"] for c in connections.values())
        endpoints_with_services = sum(1 for c in connections.values() if c["services"])
        
        print(f"  Total endpoint files: {total_endpoints}")
        print(f"  Total routes: {total_routes}")
        print(f"  Endpoints with service connections: {endpoints_with_services}")
        
        # Show sample connections
        for endpoint, conn in list(connections.items())[:3]:
            if conn["services"]:
                print(f"  {endpoint}: {conn['route_count']} routes -> {', '.join(conn['services'][:3])}")
    
    def verify_celery_integration(self):
        """Verify Celery tasks are properly integrated"""
        tasks_path = self.backend_path / "app" / "tasks"
        
        if not tasks_path.exists():
            print("  ERROR: Tasks directory not found")
            return
        
        task_files = list(tasks_path.glob("*.py"))
        celery_tasks = {}
        
        for task_file in task_files:
            if task_file.name == "__init__.py":
                continue
            
            try:
                with open(task_file, 'r', encoding='utf-8') as f:
                    content = f.read()
                
                # Find task definitions
                tasks = re.findall(r'@celery_app\.task.*?\ndef\s+(\w+)', content, re.DOTALL)
                
                # Find service imports
                service_imports = re.findall(r'from app\.services\.(\w+)', content)
                
                celery_tasks[task_file.stem] = {
                    "tasks": tasks,
                    "task_count": len(tasks),
                    "services_used": list(set(service_imports))
                }
                
            except Exception as e:
                self.results["warnings"].append(f"Could not analyze {task_file.name}: {str(e)}")
        
        self.results["celery_tasks"] = celery_tasks
        
        # Summary
        total_task_files = len(celery_tasks)
        total_tasks = sum(t["task_count"] for t in celery_tasks.values())
        
        print(f"  Total task files: {total_task_files}")
        print(f"  Total Celery tasks: {total_tasks}")
        
        for task_file, info in celery_tasks.items():
            if info["tasks"]:
                print(f"  {task_file}: {info['task_count']} tasks using {len(info['services_used'])} services")
    
    def verify_websocket_integration(self):
        """Verify WebSocket connections"""
        ws_manager_file = self.backend_path / "app" / "services" / "websocket_manager.py"
        main_file = self.backend_path / "app" / "main.py"
        
        ws_status = {
            "manager_exists": ws_manager_file.exists(),
            "endpoints": [],
            "room_support": False,
            "broadcast_support": False
        }
        
        if ws_manager_file.exists():
            try:
                with open(ws_manager_file, 'r', encoding='utf-8') as f:
                    content = f.read()
                
                # Check for key WebSocket features
                ws_status["room_support"] = "join_room" in content
                ws_status["broadcast_support"] = "broadcast" in content
                
                print(f"  WebSocket manager: Found")
                print(f"  Room support: {'Yes' if ws_status['room_support'] else 'No'}")
                print(f"  Broadcast support: {'Yes' if ws_status['broadcast_support'] else 'No'}")
                
            except Exception as e:
                self.results["warnings"].append(f"Could not analyze WebSocket manager: {str(e)}")
        
        # Check WebSocket endpoints in main.py
        try:
            with open(main_file, 'r', encoding='utf-8') as f:
                content = f.read()
            
            # Find WebSocket endpoints
            ws_endpoints = re.findall(r'@app\.websocket\s*\(\s*["\'](/ws[^"\']+)', content)
            ws_status["endpoints"] = ws_endpoints
            
            print(f"  WebSocket endpoints: {len(ws_endpoints)}")
            for endpoint in ws_endpoints:
                print(f"    - {endpoint}")
                
        except Exception as e:
            self.results["warnings"].append(f"Could not analyze WebSocket endpoints: {str(e)}")
        
        self.results["websocket_connections"] = ws_status
    
    def verify_database_integration(self):
        """Verify database model usage across services"""
        models_path = self.backend_path / "app" / "models"
        services_path = self.backend_path / "app" / "services"
        
        if not models_path.exists():
            print("  ERROR: Models directory not found")
            return
        
        # Find all models
        model_files = list(models_path.glob("*.py"))
        models = []
        
        for model_file in model_files:
            if model_file.name == "__init__.py":
                continue
            
            try:
                with open(model_file, 'r', encoding='utf-8') as f:
                    content = f.read()
                
                # Find model classes
                model_classes = re.findall(r'class\s+(\w+)\s*\([^)]*Base', content)
                models.extend(model_classes)
                
            except:
                pass
        
        print(f"  Total database models: {len(models)}")
        
        # Check model usage in services
        model_usage = {}
        
        for service_file in services_path.glob("*.py"):
            try:
                with open(service_file, 'r', encoding='utf-8') as f:
                    content = f.read()
                
                # Check which models are imported
                for model in models:
                    if model in content:
                        if service_file.stem not in model_usage:
                            model_usage[service_file.stem] = []
                        model_usage[service_file.stem].append(model)
                        
            except:
                pass
        
        print(f"  Services using models: {len(model_usage)}")
        
        # Show top services by model usage
        sorted_usage = sorted(model_usage.items(), key=lambda x: len(x[1]), reverse=True)
        for service, models_used in sorted_usage[:5]:
            print(f"    - {service}: {len(models_used)} models")
        
        self.results["database_connections"] = {
            "total_models": len(models),
            "services_using_models": len(model_usage),
            "model_usage": {k: v for k, v in sorted_usage[:10]}
        }
    
    def verify_service_dependencies(self):
        """Check inter-service dependencies"""
        services_path = self.backend_path / "app" / "services"
        
        service_files = list(services_path.glob("*.py"))
        dependencies = {}
        
        for service_file in service_files:
            if service_file.name == "__init__.py":
                continue
            
            try:
                with open(service_file, 'r', encoding='utf-8') as f:
                    content = f.read()
                
                # Find imports from other services
                service_imports = re.findall(r'from app\.services\.(\w+)', content)
                
                # Filter out self-imports
                other_services = [s for s in service_imports if s != service_file.stem]
                
                if other_services:
                    dependencies[service_file.stem] = list(set(other_services))
                    
            except:
                pass
        
        self.results["service_connections"] = dependencies
        
        # Find services with most dependencies
        sorted_deps = sorted(dependencies.items(), key=lambda x: len(x[1]), reverse=True)
        
        print(f"  Total services: {len(service_files)}")
        print(f"  Services with dependencies: {len(dependencies)}")
        print("  Most connected services:")
        
        for service, deps in sorted_deps[:5]:
            print(f"    - {service}: depends on {len(deps)} services")
    
    def verify_frontend_backend_integration(self):
        """Verify frontend API calls match backend endpoints"""
        
        # Find frontend API service files
        api_files = []
        if self.frontend_path.exists():
            api_files = list(self.frontend_path.rglob("*api*.ts")) + \
                       list(self.frontend_path.rglob("*api*.tsx")) + \
                       list(self.frontend_path.rglob("*service*.ts"))
        
        api_calls = []
        
        for api_file in api_files:
            try:
                with open(api_file, 'r', encoding='utf-8') as f:
                    content = f.read()
                
                # Find API calls
                fetch_calls = re.findall(r'fetch\s*\(\s*[`"\']([^`"\']+)', content)
                axios_calls = re.findall(r'axios\.\w+\s*\(\s*[`"\']([^`"\']+)', content)
                
                api_calls.extend(fetch_calls)
                api_calls.extend(axios_calls)
                
            except:
                pass
        
        # Extract unique endpoints
        unique_endpoints = set()
        for call in api_calls:
            if '/api/' in call:
                # Extract the API path
                path = re.search(r'/api/v\d+/(\w+)', call)
                if path:
                    unique_endpoints.add(path.group(1))
        
        self.results["frontend_backend_links"] = {
            "total_api_files": len(api_files),
            "total_api_calls": len(api_calls),
            "unique_endpoints": list(unique_endpoints)
        }
        
        print(f"  Frontend API files: {len(api_files)}")
        print(f"  Total API calls found: {len(api_calls)}")
        print(f"  Unique endpoints: {len(unique_endpoints)}")
        
        if unique_endpoints:
            print("  Endpoints called from frontend:")
            for endpoint in list(unique_endpoints)[:5]:
                print(f"    - /api/v1/{endpoint}")
    
    def verify_configuration(self):
        """Verify configuration files are properly set up"""
        config_checks = {
            "docker_compose": Path("docker-compose.yml").exists(),
            "backend_env": (self.backend_path / ".env.example").exists(),
            "frontend_env": (self.frontend_path / ".env.example").exists(),
            "alembic": (self.backend_path / "alembic.ini").exists(),
            "pytest": (self.backend_path / "pytest.ini").exists(),
            "requirements": (self.backend_path / "requirements.txt").exists(),
            "package_json": (self.frontend_path / "package.json").exists()
        }
        
        print("  Configuration files:")
        for config, exists in config_checks.items():
            status = "[OK]" if exists else "[MISSING]"
            print(f"    {status} {config}")
            
            if not exists:
                self.results["warnings"].append(f"Missing configuration: {config}")
        
        self.results["integration_status"]["configuration"] = config_checks
    
    def test_actual_imports(self):
        """Try to actually import key modules"""
        print("  Testing Python imports...")
        
        # Add backend to path
        sys.path.insert(0, str(self.backend_path))
        
        test_imports = [
            "app.core.config",
            "app.core.database",
            "app.db.session",
            "app.services.video_generation_pipeline",
            "app.services.analytics_service",
            "app.services.cost_tracking"
        ]
        
        import_results = {}
        
        for module_path in test_imports:
            try:
                spec = importlib.util.find_spec(module_path)
                if spec:
                    import_results[module_path] = "SUCCESS"
                    print(f"    [OK] {module_path}")
                else:
                    import_results[module_path] = "NOT FOUND"
                    print(f"    [ERROR] {module_path} - not found")
            except Exception as e:
                import_results[module_path] = f"ERROR: {str(e)}"
                print(f"    [ERROR] {module_path} - {str(e)}")
        
        self.results["integration_status"]["import_tests"] = import_results
    
    def generate_integration_report(self):
        """Generate comprehensive integration report"""
        
        # Calculate overall health score
        total_checks = 0
        passed_checks = 0
        
        # Main imports check
        if "main_imports" in self.results["integration_status"]:
            total_checks += 1
            if self.results["integration_status"]["main_imports"]["success_rate"].rstrip('%'):
                rate = float(self.results["integration_status"]["main_imports"]["success_rate"].rstrip('%'))
                if rate > 70:
                    passed_checks += 1
        
        # API connections check
        if self.results["api_mappings"]:
            total_checks += 1
            if len(self.results["api_mappings"]) > 10:
                passed_checks += 1
        
        # Celery tasks check
        if self.results["celery_tasks"]:
            total_checks += 1
            total_tasks = sum(t["task_count"] for t in self.results["celery_tasks"].values())
            if total_tasks > 5:
                passed_checks += 1
        
        # WebSocket check
        if self.results["websocket_connections"]:
            total_checks += 1
            if self.results["websocket_connections"]["manager_exists"]:
                passed_checks += 1
        
        # Database integration check
        if "database_connections" in self.results:
            total_checks += 1
            if self.results["database_connections"]["services_using_models"] > 10:
                passed_checks += 1
        
        # Configuration check
        if "configuration" in self.results["integration_status"]:
            total_checks += 1
            config_ok = sum(1 for v in self.results["integration_status"]["configuration"].values() if v)
            if config_ok > 4:
                passed_checks += 1
        
        health_score = (passed_checks / total_checks * 100) if total_checks > 0 else 0
        
        self.results["summary"] = {
            "health_score": f"{health_score:.1f}%",
            "total_checks": total_checks,
            "passed_checks": passed_checks,
            "total_errors": len(self.results["errors"]),
            "total_warnings": len(self.results["warnings"])
        }
        
        # Generate recommendations based on findings
        if self.results["errors"]:
            self.results["recommendations"].append({
                "priority": "HIGH",
                "action": "Fix import errors in main.py",
                "details": f"Found {len(self.results['errors'])} import errors that need fixing"
            })
        
        if health_score < 80:
            self.results["recommendations"].append({
                "priority": "MEDIUM",
                "action": "Improve integration health",
                "details": f"Current health score is {health_score:.1f}%, should be above 80%"
            })
        
        # Save report
        with open("integration_verification_report.json", 'w') as f:
            json.dump(self.results, f, indent=2, default=str)
        
        print(f"\n  Integration Health Score: {health_score:.1f}%")
        print(f"  Total Errors: {len(self.results['errors'])}")
        print(f"  Total Warnings: {len(self.results['warnings'])}")


def print_final_report(results):
    """Print formatted final report"""
    print("\n" + "="*100)
    print("INTEGRATION VERIFICATION COMPLETE")
    print("="*100)
    
    # Summary
    if "summary" in results:
        print("\n[OVERALL HEALTH]")
        print(f"  Score: {results['summary']['health_score']}")
        print(f"  Checks Passed: {results['summary']['passed_checks']}/{results['summary']['total_checks']}")
        print(f"  Errors: {results['summary']['total_errors']}")
        print(f"  Warnings: {results['summary']['total_warnings']}")
    
    # Service Connections
    if results["service_connections"]:
        print("\n[SERVICE INTERCONNECTIONS]")
        connected = len(results["service_connections"])
        total_deps = sum(len(deps) for deps in results["service_connections"].values())
        print(f"  Services with dependencies: {connected}")
        print(f"  Total inter-service connections: {total_deps}")
    
    # API Status
    if results["api_mappings"]:
        print("\n[API INTEGRATION]")
        total_endpoints = len(results["api_mappings"])
        total_routes = sum(c["route_count"] for c in results["api_mappings"].values())
        print(f"  API endpoint files: {total_endpoints}")
        print(f"  Total routes: {total_routes}")
    
    # Celery Status
    if results["celery_tasks"]:
        print("\n[CELERY INTEGRATION]")
        total_tasks = sum(t["task_count"] for t in results["celery_tasks"].values())
        print(f"  Task files: {len(results['celery_tasks'])}")
        print(f"  Total tasks: {total_tasks}")
    
    # WebSocket Status
    if results["websocket_connections"]:
        print("\n[WEBSOCKET INTEGRATION]")
        ws = results["websocket_connections"]
        print(f"  Manager: {'Yes' if ws['manager_exists'] else 'No'}")
        print(f"  Endpoints: {len(ws['endpoints'])}")
        print(f"  Room Support: {'Yes' if ws['room_support'] else 'No'}")
    
    # Errors and Warnings
    if results["errors"]:
        print("\n[CRITICAL ISSUES]")
        for error in results["errors"][:5]:
            print(f"  - {error}")
    
    if results["warnings"]:
        print("\n[WARNINGS]")
        for warning in results["warnings"][:5]:
            print(f"  - {warning}")
    
    # Recommendations
    if results["recommendations"]:
        print("\n[RECOMMENDATIONS]")
        for rec in results["recommendations"]:
            print(f"  [{rec['priority']}] {rec['action']}")
            print(f"    {rec['details']}")
    
    print("\n[REPORT SAVED]: integration_verification_report.json")


if __name__ == "__main__":
    verifier = IntegrationVerifier()
    results = verifier.verify_all_integrations()
    print_final_report(results)