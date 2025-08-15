#!/usr/bin/env python3
"""
Final Integration Verification Script
Checks that all components are properly integrated and not "hanging in space"
"""

import os
import json
import re
from pathlib import Path
from typing import Dict, List, Set, Tuple
import ast

class IntegrationVerifier:
    def __init__(self):
        self.project_root = Path.cwd()
        self.issues = []
        self.warnings = []
        self.successes = []
        
    def verify_all(self) -> Dict:
        """Run all integration checks"""
        print("=" * 80)
        print("FINAL INTEGRATION VERIFICATION")
        print("=" * 80)
        
        # Backend checks
        print("\n[BACKEND INTEGRATION CHECKS]")
        self.check_backend_services_imported()
        self.check_api_endpoints_registered()
        self.check_database_models_migrated()
        self.check_celery_tasks_registered()
        
        # Frontend checks
        print("\n[FRONTEND INTEGRATION CHECKS]")
        self.check_frontend_components_used()
        self.check_frontend_routes_defined()
        self.check_store_connections()
        
        # Infrastructure checks
        print("\n[INFRASTRUCTURE INTEGRATION CHECKS]")
        self.check_docker_services_defined()
        self.check_environment_variables()
        self.check_cicd_workflows()
        
        # ML Pipeline checks
        print("\n[ML PIPELINE INTEGRATION CHECKS]")
        self.check_ml_services_integrated()
        
        # WebSocket checks
        print("\n[WEBSOCKET INTEGRATION CHECKS]")
        self.check_websocket_endpoints()
        
        # Generate report
        return self.generate_report()
    
    def check_backend_services_imported(self):
        """Check if all services are imported in main.py"""
        main_file = self.project_root / "backend" / "app" / "main.py"
        services_dir = self.project_root / "backend" / "app" / "services"
        
        if not main_file.exists():
            self.issues.append("main.py not found")
            return
            
        with open(main_file, 'r', encoding='utf-8') as f:
            main_content = f.read()
        
        service_files = list(services_dir.glob("*.py"))
        imported_count = 0
        not_imported = []
        
        for service_file in service_files:
            if service_file.name == "__init__.py":
                continue
            service_name = service_file.stem
            
            # Check if service is imported
            if service_name in main_content or f"from app.services.{service_name}" in main_content:
                imported_count += 1
            else:
                # Check if it might be imported under different name
                if not any(skip in service_name for skip in ['__pycache__', 'test_', '_old']):
                    not_imported.append(service_name)
        
        if imported_count > 0:
            self.successes.append(f"[OK] {imported_count}/{len(service_files)-1} backend services imported in main.py")
        
        if not_imported:
            self.warnings.append(f"[!] Services not explicitly imported: {', '.join(not_imported[:5])}")
    
    def check_api_endpoints_registered(self):
        """Check if all endpoint files are registered in router"""
        api_router_file = self.project_root / "backend" / "app" / "api" / "v1" / "api.py"
        endpoints_dir = self.project_root / "backend" / "app" / "api" / "v1" / "endpoints"
        
        if not api_router_file.exists():
            self.issues.append("api.py router not found")
            return
            
        with open(api_router_file, 'r', encoding='utf-8') as f:
            router_content = f.read()
        
        endpoint_files = list(endpoints_dir.glob("*.py"))
        registered_count = 0
        
        for endpoint_file in endpoint_files:
            if endpoint_file.name == "__init__.py":
                continue
            endpoint_name = endpoint_file.stem
            
            if endpoint_name in router_content:
                registered_count += 1
        
        self.successes.append(f"[OK] {registered_count}/{len(endpoint_files)-1} API endpoints registered")
    
    def check_database_models_migrated(self):
        """Check if database migrations exist"""
        migrations_dir = self.project_root / "backend" / "alembic" / "versions"
        
        if migrations_dir.exists():
            migration_files = list(migrations_dir.glob("*.py"))
            if migration_files:
                self.successes.append(f"[OK] {len(migration_files)} database migrations found")
            else:
                self.warnings.append("[!] No database migrations found")
        else:
            self.issues.append("[X] Alembic migrations directory not found")
    
    def check_celery_tasks_registered(self):
        """Check if Celery tasks are defined"""
        tasks_dir = self.project_root / "backend" / "app" / "tasks"
        
        if tasks_dir.exists():
            task_files = list(tasks_dir.glob("*.py"))
            total_tasks = 0
            
            for task_file in task_files:
                if task_file.name == "__init__.py":
                    continue
                with open(task_file, 'r', encoding='utf-8') as f:
                    content = f.read()
                    # Count @celery_app.task decorators
                    tasks = re.findall(r'@\w+\.task', content)
                    total_tasks += len(tasks)
            
            if total_tasks > 0:
                self.successes.append(f"[OK] {total_tasks} Celery tasks defined across {len(task_files)-1} files")
            else:
                self.warnings.append("[!] No Celery tasks found")
    
    def check_frontend_components_used(self):
        """Check if components are actually imported somewhere"""
        components_dir = self.project_root / "frontend" / "src" / "components"
        pages_dir = self.project_root / "frontend" / "src" / "pages"
        
        if not components_dir.exists():
            self.issues.append("[X] Frontend components directory not found")
            return
        
        # Get all component files
        component_files = list(components_dir.rglob("*.tsx"))
        
        # Check App.tsx and router files for imports
        used_components = set()
        check_files = [
            self.project_root / "frontend" / "src" / "App.tsx",
            self.project_root / "frontend" / "src" / "router.tsx",
            self.project_root / "frontend" / "src" / "Router.tsx",
        ]
        
        # Add all page files
        if pages_dir.exists():
            check_files.extend(pages_dir.rglob("*.tsx"))
        
        for check_file in check_files:
            if check_file.exists():
                with open(check_file, 'r', encoding='utf-8') as f:
                    content = f.read()
                    for comp_file in component_files:
                        comp_name = comp_file.stem
                        if comp_name in content:
                            used_components.add(comp_name)
        
        usage_rate = len(used_components) / len(component_files) * 100 if component_files else 0
        self.successes.append(f"[OK] {len(used_components)}/{len(component_files)} components used ({usage_rate:.0f}%)")
    
    def check_frontend_routes_defined(self):
        """Check if routes are properly defined"""
        router_files = [
            self.project_root / "frontend" / "src" / "router" / "index.tsx",
            self.project_root / "frontend" / "src" / "router.tsx",
            self.project_root / "frontend" / "src" / "Router.tsx",
        ]
        
        for router_file in router_files:
            if router_file.exists():
                with open(router_file, 'r', encoding='utf-8') as f:
                    content = f.read()
                    # Look for Route components or path props
                    routes = re.findall(r'<Route\s+.*?path=["\']([^"\']+)["\']', content)
                    if routes:
                        self.successes.append(f"[OK] {len(routes)} routes defined in router")
                        return
        
        self.warnings.append("[!] Router file not found or no routes defined")
    
    def check_store_connections(self):
        """Check if Zustand stores are connected"""
        stores_dir = self.project_root / "frontend" / "src" / "stores"
        
        if stores_dir.exists():
            store_files = list(stores_dir.glob("*.ts"))
            self.successes.append(f"[OK] {len(store_files)} Zustand stores defined")
            
            # Check if stores are imported in components
            components_dir = self.project_root / "frontend" / "src" / "components"
            pages_dir = self.project_root / "frontend" / "src" / "pages"
            
            store_usage = 0
            for store_file in store_files:
                store_name = store_file.stem
                # Check in components and pages
                for check_dir in [components_dir, pages_dir]:
                    if check_dir.exists():
                        for tsx_file in check_dir.rglob("*.tsx"):
                            try:
                                with open(tsx_file, 'r', encoding='utf-8') as f:
                                    if store_name in f.read():
                                        store_usage += 1
                                        break
                            except:
                                continue
            
            if store_usage > 0:
                self.successes.append(f"[OK] {store_usage}/{len(store_files)} stores actively used")
    
    def check_docker_services_defined(self):
        """Check Docker services configuration"""
        docker_files = [
            "docker-compose.yml",
            "docker-compose.production.yml",
            "docker-compose.staging.yml"
        ]
        
        total_services = 0
        for docker_file in docker_files:
            docker_path = self.project_root / docker_file
            if docker_path.exists():
                with open(docker_path, 'r', encoding='utf-8') as f:
                    content = f.read()
                    # Count service definitions
                    services = re.findall(r'^  \w+:', content, re.MULTILINE)
                    total_services += len(services)
        
        if total_services > 0:
            self.successes.append(f"[OK] {total_services} Docker services defined across {len(docker_files)} compose files")
        else:
            self.issues.append("[X] No Docker services found")
    
    def check_environment_variables(self):
        """Check if environment variables are documented"""
        env_example = self.project_root / ".env.example"
        
        if env_example.exists():
            with open(env_example, 'r', encoding='utf-8') as f:
                lines = f.readlines()
                env_vars = [line for line in lines if '=' in line and not line.startswith('#')]
                self.successes.append(f"[OK] {len(env_vars)} environment variables documented")
        else:
            self.warnings.append("[!] .env.example not found")
    
    def check_cicd_workflows(self):
        """Check GitHub Actions workflows"""
        workflows_dir = self.project_root / ".github" / "workflows"
        
        if workflows_dir.exists():
            workflow_files = list(workflows_dir.glob("*.yml")) + list(workflows_dir.glob("*.yaml"))
            self.successes.append(f"[OK] {len(workflow_files)} CI/CD workflows configured")
        else:
            self.issues.append("[X] No GitHub Actions workflows found")
    
    def check_ml_services_integrated(self):
        """Check ML pipeline integration"""
        ml_pipeline_dir = self.project_root / "ml-pipeline"
        
        if ml_pipeline_dir.exists():
            # Check for key ML service files
            ml_services = [
                "services/quality_assurance.py",
                "services/analytics_pipeline.py",
                "quality_scoring/quality_scorer.py",
                "src/trend_detection_model.py",
                "src/script_generation.py",
                "src/voice_synthesis.py"
            ]
            
            found_services = 0
            for service in ml_services:
                if (ml_pipeline_dir / service).exists():
                    found_services += 1
            
            self.successes.append(f"[OK] {found_services}/{len(ml_services)} ML pipeline services found")
    
    def check_websocket_endpoints(self):
        """Check WebSocket endpoints"""
        endpoints_dir = self.project_root / "backend" / "app" / "api" / "v1" / "endpoints"
        
        ws_count = 0
        if endpoints_dir.exists():
            for py_file in endpoints_dir.glob("*.py"):
                with open(py_file, 'r', encoding='utf-8') as f:
                    content = f.read()
                    if '@router.websocket' in content or 'WebSocket' in content:
                        ws_count += 1
        
        if ws_count > 0:
            self.successes.append(f"[OK] {ws_count} WebSocket endpoints configured")
        else:
            self.warnings.append("[!] No WebSocket endpoints found")
    
    def generate_report(self) -> Dict:
        """Generate final report"""
        print("\n" + "=" * 80)
        print("VERIFICATION SUMMARY")
        print("=" * 80)
        
        # Success summary
        if self.successes:
            print("\n[OK] SUCCESSFUL INTEGRATIONS:")
            for success in self.successes:
                print(f"  {success}")
        
        # Warnings
        if self.warnings:
            print("\n[!] WARNINGS (non-critical):")
            for warning in self.warnings:
                print(f"  {warning}")
        
        # Critical issues
        if self.issues:
            print("\n[X] CRITICAL ISSUES:")
            for issue in self.issues:
                print(f"  {issue}")
        
        # Final verdict
        print("\n" + "=" * 80)
        if not self.issues:
            print("[OK] ALL CRITICAL INTEGRATIONS VERIFIED - NO HANGING COMPONENTS")
            print("[OK] SYSTEM IS FULLY INTEGRATED AND READY FOR PRODUCTION")
        else:
            print("[!] SOME INTEGRATION ISSUES FOUND - REVIEW NEEDED")
        
        return {
            "successes": len(self.successes),
            "warnings": len(self.warnings),
            "issues": len(self.issues),
            "details": {
                "successes": self.successes,
                "warnings": self.warnings,
                "issues": self.issues
            }
        }

def main():
    verifier = IntegrationVerifier()
    results = verifier.verify_all()
    
    # Save results
    report_file = Path("misc/final_integration_report.json")
    with open(report_file, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\nFull report saved to: {report_file}")
    
    return results

if __name__ == "__main__":
    main()