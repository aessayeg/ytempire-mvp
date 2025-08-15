"""
Comprehensive Project Audit - Week 2 Complete
Verifies ALL expected features and identifies duplicates
"""

import os
import json
import re
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Set, Tuple
from collections import defaultdict

class ComprehensiveAuditor:
    def __init__(self):
        self.project_root = Path(".")
        self.results = {
            "timestamp": datetime.now().isoformat(),
            "backend_services": {},
            "frontend_components": {},
            "api_endpoints": {},
            "tasks_completed": {},
            "duplicates": {},
            "missing_features": [],
            "recommendations": []
        }
        
    def audit_everything(self):
        """Complete project audit"""
        print("\n" + "="*100)
        print("COMPREHENSIVE PROJECT AUDIT - WEEK 2 COMPLETION VERIFICATION")
        print("="*100)
        
        # 1. Backend Services Deep Scan
        print("\n[1/7] BACKEND SERVICES AUDIT...")
        self.audit_backend_services()
        
        # 2. Frontend Components Deep Scan  
        print("\n[2/7] FRONTEND COMPONENTS AUDIT...")
        self.audit_frontend_components()
        
        # 3. API Endpoints Verification
        print("\n[3/7] API ENDPOINTS AUDIT...")
        self.audit_api_endpoints()
        
        # 4. Celery Tasks Verification
        print("\n[4/7] CELERY TASKS AUDIT...")
        self.audit_celery_tasks()
        
        # 5. Database Models Check
        print("\n[5/7] DATABASE MODELS AUDIT...")
        self.audit_database_models()
        
        # 6. Week 2 Task Completion Verification
        print("\n[6/7] WEEK 2 TASK COMPLETION VERIFICATION...")
        self.verify_week2_tasks()
        
        # 7. Duplicate Analysis
        print("\n[7/7] DUPLICATE & REDUNDANCY ANALYSIS...")
        self.find_all_duplicates()
        
        return self.results
    
    def audit_backend_services(self):
        """Deep scan of all backend services"""
        services_path = Path("backend/app/services")
        
        if not services_path.exists():
            print("  [ERROR] Backend services directory not found!")
            return
        
        all_services = list(services_path.glob("*.py"))
        print(f"  Found {len(all_services)} service files")
        
        # Categorize services
        service_categories = {
            "Video Generation": [],
            "YouTube/Channel": [],
            "Analytics": [],
            "Payment/Billing": [],
            "AI/ML": [],
            "Notification": [],
            "Cost Tracking": [],
            "Authentication": [],
            "Data Processing": [],
            "WebSocket/Realtime": [],
            "Batch Processing": [],
            "Other": []
        }
        
        for service_file in all_services:
            if service_file.name == "__init__.py":
                continue
                
            name = service_file.stem
            size = service_file.stat().st_size
            
            # Read file to get functions/classes
            content_info = self.analyze_service_file(service_file)
            
            # Categorize
            if any(x in name.lower() for x in ["video", "generation", "pipeline", "orchestrat"]):
                service_categories["Video Generation"].append({
                    "name": name, "size": size, "info": content_info
                })
            elif any(x in name.lower() for x in ["youtube", "channel"]):
                service_categories["YouTube/Channel"].append({
                    "name": name, "size": size, "info": content_info
                })
            elif any(x in name.lower() for x in ["analytics", "metric", "report"]):
                service_categories["Analytics"].append({
                    "name": name, "size": size, "info": content_info
                })
            elif any(x in name.lower() for x in ["payment", "billing", "subscription", "invoice"]):
                service_categories["Payment/Billing"].append({
                    "name": name, "size": size, "info": content_info
                })
            elif any(x in name.lower() for x in ["ai", "openai", "script", "thumbnail", "elevenlabs"]):
                service_categories["AI/ML"].append({
                    "name": name, "size": size, "info": content_info
                })
            elif any(x in name.lower() for x in ["notification", "alert", "email"]):
                service_categories["Notification"].append({
                    "name": name, "size": size, "info": content_info
                })
            elif any(x in name.lower() for x in ["cost", "tracking", "budget"]):
                service_categories["Cost Tracking"].append({
                    "name": name, "size": size, "info": content_info
                })
            elif any(x in name.lower() for x in ["auth", "security", "oauth"]):
                service_categories["Authentication"].append({
                    "name": name, "size": size, "info": content_info
                })
            elif any(x in name.lower() for x in ["websocket", "ws", "realtime", "room"]):
                service_categories["WebSocket/Realtime"].append({
                    "name": name, "size": size, "info": content_info
                })
            elif any(x in name.lower() for x in ["batch", "bulk", "queue"]):
                service_categories["Batch Processing"].append({
                    "name": name, "size": size, "info": content_info
                })
            elif any(x in name.lower() for x in ["data", "etl", "pipeline", "processing"]):
                service_categories["Data Processing"].append({
                    "name": name, "size": size, "info": content_info
                })
            else:
                service_categories["Other"].append({
                    "name": name, "size": size, "info": content_info
                })
        
        # Print summary
        for category, services in service_categories.items():
            if services:
                print(f"\n  {category}: {len(services)} services")
                for svc in services:
                    print(f"    - {svc['name']}.py ({svc['size']:,} bytes) | Classes: {svc['info']['classes']}, Functions: {svc['info']['functions']}")
        
        self.results["backend_services"] = service_categories
    
    def analyze_service_file(self, file_path):
        """Analyze a service file for classes and functions"""
        try:
            with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                content = f.read()
                
            # Extract classes
            classes = re.findall(r'class\s+(\w+)', content)
            # Extract functions
            functions = re.findall(r'def\s+(\w+)\s*\(', content)
            # Extract async functions
            async_funcs = re.findall(r'async\s+def\s+(\w+)\s*\(', content)
            
            return {
                "classes": len(classes),
                "functions": len(functions),
                "async_functions": len(async_funcs),
                "class_names": classes[:3],  # First 3 classes
                "main_functions": [f for f in functions if not f.startswith('_')][:5]
            }
        except:
            return {"classes": 0, "functions": 0, "async_functions": 0}
    
    def audit_frontend_components(self):
        """Deep scan of frontend components"""
        components_path = Path("frontend/src/components")
        pages_path = Path("frontend/src/pages")
        
        components_found = {}
        
        # Scan components
        if components_path.exists():
            all_components = list(components_path.rglob("*.tsx")) + list(components_path.rglob("*.jsx"))
            print(f"  Found {len(all_components)} component files")
            
            # Categorize
            categories = defaultdict(list)
            for comp in all_components:
                rel_path = comp.relative_to(components_path)
                parent = rel_path.parts[0] if len(rel_path.parts) > 1 else "Root"
                categories[parent].append(str(rel_path))
            
            for category, files in sorted(categories.items()):
                print(f"    {category}: {len(files)} components")
                components_found[category] = files
        
        # Scan pages
        if pages_path.exists():
            all_pages = list(pages_path.rglob("*.tsx")) + list(pages_path.rglob("*.jsx"))
            print(f"\n  Found {len(all_pages)} page files")
            
            pages_found = []
            for page in all_pages:
                rel_path = page.relative_to(pages_path)
                print(f"    - {rel_path}")
                pages_found.append(str(rel_path))
            
            components_found["Pages"] = pages_found
        
        self.results["frontend_components"] = components_found
    
    def audit_api_endpoints(self):
        """Verify all API endpoints"""
        endpoints_path = Path("backend/app/api/v1/endpoints")
        
        if not endpoints_path.exists():
            print("  [ERROR] API endpoints directory not found!")
            return
        
        endpoint_files = list(endpoints_path.glob("*.py"))
        print(f"  Found {len(endpoint_files)} endpoint files")
        
        endpoints = {}
        for file in endpoint_files:
            if file.name == "__init__.py":
                continue
            
            # Analyze endpoint file
            try:
                with open(file, 'r', encoding='utf-8', errors='ignore') as f:
                    content = f.read()
                
                # Find route decorators
                routes = re.findall(r'@router\.(get|post|put|delete|patch)\s*\(\s*["\']([^"\']+)', content)
                
                endpoints[file.stem] = {
                    "file": file.name,
                    "routes": routes,
                    "count": len(routes)
                }
                
                print(f"    - {file.stem}: {len(routes)} routes")
            except:
                print(f"    - {file.stem}: [ERROR reading]")
        
        self.results["api_endpoints"] = endpoints
    
    def audit_celery_tasks(self):
        """Verify Celery task files"""
        tasks_path = Path("backend/app/tasks")
        
        if not tasks_path.exists():
            print("  [ERROR] Tasks directory not found!")
            return
        
        task_files = list(tasks_path.glob("*.py"))
        print(f"  Found {len(task_files)} task files")
        
        tasks = {}
        for file in task_files:
            if file.name == "__init__.py":
                continue
            
            try:
                with open(file, 'r', encoding='utf-8', errors='ignore') as f:
                    content = f.read()
                
                # Find Celery tasks
                celery_tasks = re.findall(r'@celery_app\.task|@app\.task|@task', content)
                
                tasks[file.stem] = {
                    "file": file.name,
                    "task_count": len(celery_tasks),
                    "size": file.stat().st_size
                }
                
                print(f"    - {file.stem}: {len(celery_tasks)} tasks ({file.stat().st_size:,} bytes)")
            except:
                print(f"    - {file.stem}: [ERROR reading]")
        
        self.results["celery_tasks"] = tasks
    
    def audit_database_models(self):
        """Check database models"""
        models_path = Path("backend/app/models")
        
        if not models_path.exists():
            print("  [ERROR] Models directory not found!")
            return
        
        model_files = list(models_path.glob("*.py"))
        print(f"  Found {len(model_files)} model files")
        
        models = {}
        for file in model_files:
            if file.name == "__init__.py":
                continue
            
            try:
                with open(file, 'r', encoding='utf-8', errors='ignore') as f:
                    content = f.read()
                
                # Find SQLAlchemy models
                model_classes = re.findall(r'class\s+(\w+)\s*\([^)]*Base[^)]*\)', content)
                
                models[file.stem] = {
                    "file": file.name,
                    "models": model_classes,
                    "count": len(model_classes)
                }
                
                print(f"    - {file.stem}: {len(model_classes)} models ({', '.join(model_classes[:3])}...)")
            except:
                print(f"    - {file.stem}: [ERROR reading]")
        
        self.results["database_models"] = models
    
    def verify_week2_tasks(self):
        """Verify Week 2 expected deliverables based on documentation"""
        
        # Based on Week 2 Master Plan
        expected_features = {
            "Backend P0": {
                "Multi-Channel Architecture": [
                    "backend/app/services/youtube_multi_account.py",
                    "backend/app/services/channel_manager.py",
                    "backend/app/models/channel.py"
                ],
                "Batch Processing (50-100 videos/day)": [
                    "backend/app/services/batch_processing.py",
                    "backend/app/tasks/batch_tasks.py",
                    "backend/app/api/v1/endpoints/batch.py"
                ],
                "Subscription System": [
                    "backend/app/services/subscription_service.py",
                    "backend/app/services/payment_service",  # Could be enhanced version
                    "backend/app/models/subscription.py"
                ],
                "WebSocket Real-time": [
                    "backend/app/services/websocket_manager.py",
                    "backend/app/websocket",  # Directory
                ],
                "Cost Optimization (<$2/video)": [
                    "backend/app/services/cost_tracking.py",
                    "backend/app/services/realtime_cost_tracking.py"  # Might exist
                ],
                "Database Optimization (200 connections)": [
                    "backend/app/core/database.py"  # Should have pool configuration
                ]
            },
            "Frontend P0": {
                "Channel Dashboard": [
                    "frontend/src/components/Dashboard",
                    "frontend/src/components/Channels",
                    "frontend/src/pages/Dashboard"
                ],
                "Real-time Monitoring": [
                    "frontend/src/components/Monitoring",
                    "frontend/src/components/Analytics/RealtimeAnalytics"
                ],
                "Mobile Responsive": [
                    # Check for responsive components
                ],
                "Beta User Onboarding": [
                    "frontend/src/components/Onboarding",
                    "frontend/src/pages/Onboarding"
                ]
            },
            "AI/ML P0": {
                "Multi-Model Orchestration": [
                    "backend/app/services/ai_services.py",
                    "ml-pipeline/services"
                ],
                "Cost Optimization": [
                    # Should be in AI services
                ],
                "Personalization Engine": [
                    "backend/app/services/personalization",
                    "ml-pipeline/models/personalization"
                ]
            },
            "Data P0": {
                "Real-time Analytics Pipeline": [
                    "backend/app/services/analytics_pipeline.py",
                    "backend/app/services/realtime_analytics_service.py"
                ],
                "Beta User Analytics": [
                    "backend/app/services/analytics_service.py"
                ]
            },
            "Platform Ops P0": {
                "Production Deployment": [
                    "docker-compose.yml",
                    "infrastructure/kubernetes",
                    ".github/workflows"
                ],
                "Monitoring": [
                    "infrastructure/monitoring",
                    "docker-compose.monitoring.yml"
                ],
                "Security Hardening": [
                    "backend/app/core/security.py",
                    "infrastructure/security"
                ]
            }
        }
        
        completion_status = {}
        
        for category, features in expected_features.items():
            print(f"\n  {category}:")
            category_status = {}
            
            for feature, expected_files in features.items():
                found_count = 0
                total_count = len(expected_files)
                found_files = []
                missing_files = []
                
                for expected_file in expected_files:
                    if Path(expected_file).exists():
                        found_count += 1
                        found_files.append(expected_file)
                    else:
                        # Try alternative names
                        alternatives = self.find_alternative_files(expected_file)
                        if alternatives:
                            found_count += 1
                            found_files.extend(alternatives)
                        else:
                            missing_files.append(expected_file)
                
                completion = (found_count / total_count * 100) if total_count > 0 else 0
                status = "[COMPLETE]" if completion == 100 else f"[{completion:.0f}%]"
                
                print(f"    {feature}: {status}")
                if missing_files:
                    print(f"      Missing: {', '.join(missing_files)}")
                
                category_status[feature] = {
                    "completion": completion,
                    "found": found_files,
                    "missing": missing_files
                }
            
            completion_status[category] = category_status
        
        self.results["tasks_completed"] = completion_status
    
    def find_alternative_files(self, expected_path):
        """Find files with similar names"""
        path = Path(expected_path)
        parent = path.parent
        name = path.stem
        
        if not parent.exists():
            return []
        
        # Look for similar files
        alternatives = []
        
        # Common patterns
        patterns = [
            f"*{name}*",
            f"*{'_'.join(name.split('_')[:2])}*" if '_' in name else f"*{name[:5]}*",
            name.replace("_service", "").replace("_manager", "")
        ]
        
        for pattern in patterns:
            matches = list(parent.glob(f"{pattern}.*"))
            alternatives.extend([str(m) for m in matches if m.is_file()])
        
        return list(set(alternatives))[:3]  # Return first 3 matches
    
    def find_all_duplicates(self):
        """Find all duplicate and redundant files"""
        duplicates = defaultdict(list)
        
        # Check backend services
        if "backend_services" in self.results:
            for category, services in self.results["backend_services"].items():
                if len(services) > 1:
                    # Check for actual duplicates
                    if category == "Video Generation" and len(services) > 2:
                        duplicates["backend_video_generation"] = [s["name"] for s in services]
                    elif category == "Analytics" and len(services) > 3:
                        duplicates["backend_analytics"] = [s["name"] for s in services]
                    elif category == "Payment/Billing" and len(services) > 2:
                        # Check for enhanced versions
                        names = [s["name"] for s in services]
                        if any("enhanced" in n for n in names):
                            duplicates["backend_payment"] = names
                    elif category == "Cost Tracking" and len(services) > 1:
                        duplicates["backend_cost_tracking"] = [s["name"] for s in services]
        
        self.results["duplicates"] = dict(duplicates)
        
        if duplicates:
            print(f"\n  Found {len(duplicates)} categories with duplicates:")
            for category, files in duplicates.items():
                print(f"    {category}: {len(files)} files")
                for f in files:
                    print(f"      - {f}")


def generate_cleanup_recommendations(results):
    """Generate specific cleanup recommendations"""
    recommendations = []
    
    # Check for video generation duplicates
    if "backend_video_generation" in results.get("duplicates", {}):
        files = results["duplicates"]["backend_video_generation"]
        recommendations.append({
            "priority": "HIGH",
            "category": "Video Generation Services",
            "action": "CONSOLIDATE",
            "details": f"Found {len(files)} video generation services",
            "files_to_keep": ["video_generation_pipeline.py"],
            "files_to_merge": ["video_generation_orchestrator.py", "enhanced_video_generation.py"],
            "files_to_remove": [f for f in files if f not in ["video_generation_pipeline.py", "video_generation_orchestrator.py", "enhanced_video_generation.py"]],
            "justification": "video_generation_pipeline.py appears to be the most complete implementation with proper Celery integration"
        })
    
    # Check for analytics duplicates
    if "backend_analytics" in results.get("duplicates", {}):
        files = results["duplicates"]["backend_analytics"]
        recommendations.append({
            "priority": "HIGH",
            "category": "Analytics Services",
            "action": "CONSOLIDATE",
            "details": f"Found {len(files)} analytics services",
            "files_to_keep": ["analytics_service.py"],
            "files_to_merge": ["realtime_analytics_service.py", "analytics_pipeline.py"],
            "files_to_remove": ["analytics_connector.py", "analytics_report.py"],
            "justification": "Multiple analytics implementations create confusion. Merge realtime features into main service."
        })
    
    # Check for cost tracking duplicates
    if "backend_cost_tracking" in results.get("duplicates", {}):
        recommendations.append({
            "priority": "MEDIUM",
            "category": "Cost Tracking",
            "action": "MERGE",
            "details": "Two cost tracking services exist",
            "files_to_keep": ["cost_tracking.py"],
            "files_to_merge": ["realtime_cost_tracking.py"],
            "files_to_remove": [],
            "justification": "Realtime features should be part of main cost tracking service"
        })
    
    # Check for missing critical features
    if "tasks_completed" in results:
        for category, features in results["tasks_completed"].items():
            for feature, status in features.items():
                if status["completion"] < 100:
                    missing = status["missing"]
                    if missing:
                        recommendations.append({
                            "priority": "CRITICAL" if "P0" in category else "MEDIUM",
                            "category": f"{category} - {feature}",
                            "action": "IMPLEMENT",
                            "details": f"Feature {status['completion']:.0f}% complete",
                            "files_to_create": missing,
                            "justification": f"Required for Week 2 {category} completion"
                        })
    
    return recommendations


def print_audit_report(results):
    """Print comprehensive audit report"""
    print("\n" + "="*100)
    print("AUDIT RESULTS SUMMARY")
    print("="*100)
    
    # Backend Services Summary
    if "backend_services" in results:
        print("\n[BACKEND SERVICES SUMMARY]")
        total_services = sum(len(services) for services in results["backend_services"].values())
        print(f"Total Services: {total_services}")
        
        for category, services in results["backend_services"].items():
            if services:
                total_size = sum(s["size"] for s in services)
                print(f"  {category}: {len(services)} services ({total_size:,} bytes total)")
    
    # Task Completion Summary
    if "tasks_completed" in results:
        print("\n[WEEK 2 TASK COMPLETION]")
        for category, features in results["tasks_completed"].items():
            completed = sum(1 for f in features.values() if f["completion"] == 100)
            total = len(features)
            percentage = (completed / total * 100) if total > 0 else 0
            print(f"  {category}: {percentage:.0f}% ({completed}/{total} features)")
    
    # Duplicates Summary
    if "duplicates" in results:
        print("\n[DUPLICATES FOUND]")
        for category, files in results["duplicates"].items():
            print(f"  {category}: {len(files)} duplicate files")
    
    # Generate recommendations
    recommendations = generate_cleanup_recommendations(results)
    
    if recommendations:
        print("\n[CLEANUP RECOMMENDATIONS]")
        
        # Group by priority
        high = [r for r in recommendations if r["priority"] == "HIGH"]
        critical = [r for r in recommendations if r["priority"] == "CRITICAL"]
        medium = [r for r in recommendations if r["priority"] == "MEDIUM"]
        
        if critical:
            print("\n[CRITICAL] - Missing Required Features:")
            for rec in critical[:5]:  # Show top 5
                print(f"\n  {rec['category']}:")
                print(f"    Action: {rec['action']}")
                print(f"    Details: {rec['details']}")
                if "files_to_create" in rec:
                    print(f"    Create: {', '.join(rec['files_to_create'][:3])}")
        
        if high:
            print("\n[HIGH PRIORITY] - Consolidation Required:")
            for rec in high:
                print(f"\n  {rec['category']}:")
                print(f"    Action: {rec['action']}")
                print(f"    Keep: {', '.join(rec['files_to_keep'])}")
                if rec.get('files_to_merge'):
                    print(f"    Merge: {', '.join(rec['files_to_merge'])}")
                if rec.get('files_to_remove'):
                    print(f"    Remove: {', '.join(rec['files_to_remove'])}")
                print(f"    Reason: {rec['justification']}")
        
        if medium:
            print("\n[MEDIUM PRIORITY] - Optimization:")
            for rec in medium[:3]:  # Show top 3
                print(f"\n  {rec['category']}:")
                print(f"    Action: {rec['action']}")
                print(f"    Details: {rec['details']}")
    
    # Save results
    with open("comprehensive_audit_results.json", 'w') as f:
        json.dump(results, f, indent=2, default=str)
    
    print("\n[REPORT SAVED]: comprehensive_audit_results.json")


if __name__ == "__main__":
    auditor = ComprehensiveAuditor()
    results = auditor.audit_everything()
    print_audit_report(results)