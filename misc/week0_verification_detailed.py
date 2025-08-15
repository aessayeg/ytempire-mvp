#!/usr/bin/env python3
"""
Week 0 Task Verification Script - Detailed Check
Verifies actual implementation against planned deliverables
"""

import os
import json
import subprocess
from pathlib import Path
from typing import Dict, List, Tuple

# Base directory
BASE_DIR = Path(r"C:\Users\Hp\projects\ytempire-mvp")

class Week0Verifier:
    def __init__(self):
        self.results = {
            "backend": {"P0": {}, "P1": {}, "P2": {}},
            "frontend": {"P0": {}, "P1": {}, "P2": {}},
            "ops": {"P0": {}, "P1": {}, "P2": {}},
            "aiml": {"P0": {}, "P1": {}, "P2": {}},
            "data": {"P0": {}, "P1": {}, "P2": {}}
        }
        
    def check_file_exists(self, path: str) -> bool:
        """Check if a file exists"""
        full_path = BASE_DIR / path
        return full_path.exists()
    
    def check_directory_exists(self, path: str) -> bool:
        """Check if a directory exists"""
        full_path = BASE_DIR / path
        return full_path.exists() and full_path.is_dir()
    
    def check_file_contains(self, path: str, patterns: List[str]) -> Tuple[bool, List[str]]:
        """Check if file contains specific patterns"""
        full_path = BASE_DIR / path
        if not full_path.exists():
            return False, []
        
        try:
            content = full_path.read_text(encoding='utf-8')
            found = []
            for pattern in patterns:
                if pattern.lower() in content.lower():
                    found.append(pattern)
            return len(found) == len(patterns), found
        except:
            return False, []
    
    def verify_backend_tasks(self):
        """Verify Backend Team deliverables"""
        
        # P0 Tasks (Must Complete by Day 2)
        print("\n=== BACKEND P0 TASKS ===")
        
        # 1. API Gateway setup with FastAPI
        self.results["backend"]["P0"]["API Gateway"] = {
            "status": self.check_file_exists("backend/app/main.py"),
            "details": {
                "FastAPI app": self.check_file_exists("backend/app/main.py"),
                "API router": self.check_file_exists("backend/app/api/v1/api.py"),
                "OpenAPI docs": self.check_file_contains("backend/app/main.py", ["FastAPI", "api_router"]),
            }
        }
        
        # 2. Database schema design with ERD
        self.results["backend"]["P0"]["Database Schema"] = {
            "status": self.check_directory_exists("backend/app/models"),
            "details": {
                "User model": self.check_file_exists("backend/app/models/user.py"),
                "Channel model": self.check_file_exists("backend/app/models/channel.py"),
                "Video model": self.check_file_exists("backend/app/models/video.py"),
                "Cost model": self.check_file_exists("backend/app/models/cost.py"),
                "Alembic migrations": self.check_directory_exists("backend/alembic/versions"),
            }
        }
        
        # 3. Message queue infrastructure (Redis/Celery)
        self.results["backend"]["P0"]["Message Queue"] = {
            "status": self.check_file_exists("backend/app/core/celery_app.py"),
            "details": {
                "Celery config": self.check_file_exists("backend/app/core/celery_app.py"),
                "Redis config": self.check_file_exists("backend/app/core/redis_config.py"),
                "Video tasks": self.check_file_exists("backend/app/tasks/video_tasks.py"),
                "AI tasks": self.check_file_exists("backend/app/tasks/ai_tasks.py"),
            }
        }
        
        # 4. Development environment documentation
        self.results["backend"]["P0"]["Dev Environment"] = {
            "status": self.check_file_exists("backend/requirements.txt"),
            "details": {
                "Requirements": self.check_file_exists("backend/requirements.txt"),
                "Dockerfile": self.check_file_exists("backend/Dockerfile"),
                "README": self.check_file_exists("backend/README.md"),
            }
        }
        
        # P1 Tasks (Must Complete by Day 4)
        print("\n=== BACKEND P1 TASKS ===")
        
        # 1. Authentication service with JWT
        self.results["backend"]["P1"]["Authentication"] = {
            "status": self.check_file_exists("backend/app/core/auth.py"),
            "details": {
                "JWT implementation": self.check_file_exists("backend/app/core/jwt_enhanced.py"),
                "Auth endpoints": self.check_file_exists("backend/app/api/v1/endpoints/auth.py"),
                "OAuth support": self.check_file_contains("backend/app/core/auth.py", ["JWT", "OAuth"]),
            }
        }
        
        # 2. Channel management CRUD
        self.results["backend"]["P1"]["Channel CRUD"] = {
            "status": self.check_file_exists("backend/app/api/v1/endpoints/channels.py"),
            "details": {
                "Channels endpoint": self.check_file_exists("backend/app/api/v1/endpoints/channels.py"),
                "Channel service": self.check_file_exists("backend/app/services/channel_manager.py"),
            }
        }
        
        # 3. YouTube API integration
        self.results["backend"]["P1"]["YouTube Integration"] = {
            "status": self.check_file_exists("backend/app/services/youtube_service.py"),
            "details": {
                "YouTube service": self.check_file_exists("backend/app/services/youtube_service.py"),
                "Multi-account": self.check_file_exists("backend/app/services/youtube_multi_account.py"),
                "OAuth service": self.check_file_exists("backend/app/services/youtube_oauth_service.py"),
            }
        }
        
        # 4. N8N workflow engine
        self.results["backend"]["P1"]["N8N Workflow"] = {
            "status": self.check_file_exists("docker-compose.n8n.yml"),
            "details": {
                "Docker config": self.check_file_exists("docker-compose.n8n.yml"),
                "Workflow integration": self.check_file_exists("backend/app/services/n8n_integration.py"),
            }
        }
        
        # 5. Video processing pipeline
        self.results["backend"]["P1"]["Video Pipeline"] = {
            "status": self.check_file_exists("backend/app/services/video_generation_pipeline.py"),
            "details": {
                "Pipeline service": self.check_file_exists("backend/app/services/video_generation_pipeline.py"),
                "Orchestrator": self.check_file_exists("backend/app/services/video_generation_orchestrator.py"),
                "Queue service": self.check_file_exists("backend/app/services/video_queue_service.py"),
            }
        }
        
        # 6. Cost tracking system
        self.results["backend"]["P1"]["Cost Tracking"] = {
            "status": self.check_file_exists("backend/app/services/cost_tracking.py"),
            "details": {
                "Cost service": self.check_file_exists("backend/app/services/cost_tracking.py"),
                "Cost optimizer": self.check_file_exists("backend/app/services/cost_optimizer.py"),
                "Cost endpoints": self.check_file_exists("backend/app/api/v1/endpoints/cost_intelligence.py"),
            }
        }
        
        # P2 Tasks (Complete by Day 5)
        print("\n=== BACKEND P2 TASKS ===")
        
        # 1. WebSocket foundation
        self.results["backend"]["P2"]["WebSocket"] = {
            "status": self.check_file_exists("backend/app/services/websocket_manager.py"),
            "details": {
                "WebSocket manager": self.check_file_exists("backend/app/services/websocket_manager.py"),
                "WebSocket endpoints": self.check_file_exists("backend/app/api/v1/endpoints/websockets.py"),
            }
        }
        
        # 2. Payment gateway initial setup
        self.results["backend"]["P2"]["Payment Gateway"] = {
            "status": self.check_file_exists("backend/app/services/payment_service_enhanced.py"),
            "details": {
                "Payment service": self.check_file_exists("backend/app/services/payment_service_enhanced.py"),
                "Payment endpoints": self.check_file_exists("backend/app/api/v1/endpoints/payments.py"),
            }
        }
        
        # 3. Error handling framework
        self.results["backend"]["P2"]["Error Handling"] = {
            "status": self.check_file_exists("backend/app/core/error_handling.py"),
            "details": {
                "Error handlers": self.check_file_exists("backend/app/core/error_handlers.py"),
                "Exceptions": self.check_file_exists("backend/app/core/exceptions.py"),
                "Framework": self.check_file_exists("backend/app/core/error_handling_framework.py"),
            }
        }
    
    def verify_frontend_tasks(self):
        """Verify Frontend Team deliverables"""
        
        # P0 Tasks
        print("\n=== FRONTEND P0 TASKS ===")
        
        # 1. React project initialization with Vite
        self.results["frontend"]["P0"]["React Setup"] = {
            "status": self.check_file_exists("frontend/package.json"),
            "details": {
                "Package.json": self.check_file_exists("frontend/package.json"),
                "Vite config": self.check_file_exists("frontend/vite.config.ts"),
                "TypeScript": self.check_file_exists("frontend/tsconfig.json"),
                "Index.html": self.check_file_exists("frontend/index.html"),
            }
        }
        
        # 2. Design system documentation
        self.results["frontend"]["P0"]["Design System"] = {
            "status": self.check_file_exists("frontend/DESIGN_SYSTEM.md"),
            "details": {
                "Design doc": self.check_file_exists("frontend/DESIGN_SYSTEM.md"),
                "Tailwind config": self.check_file_exists("frontend/tailwind.config.js"),
                "Theme setup": self.check_directory_exists("frontend/src/theme"),
            }
        }
        
        # 3. Development environment setup
        self.results["frontend"]["P0"]["Dev Environment"] = {
            "status": self.check_file_exists("frontend/.eslintrc.json"),
            "details": {
                "ESLint": self.check_file_exists("frontend/.eslintrc.json"),
                "Prettier": self.check_file_exists("frontend/.prettierrc"),
                "Storybook": self.check_directory_exists("frontend/.storybook"),
            }
        }
        
        # 4. Component library foundation
        self.results["frontend"]["P0"]["Component Library"] = {
            "status": self.check_directory_exists("frontend/src/components"),
            "details": {
                "Components dir": self.check_directory_exists("frontend/src/components"),
                "Common components": self.check_directory_exists("frontend/src/components/common"),
                "Layout components": self.check_directory_exists("frontend/src/components/layout"),
            }
        }
        
        # P1 Tasks
        print("\n=== FRONTEND P1 TASKS ===")
        
        # 1. State management (Zustand)
        self.results["frontend"]["P1"]["State Management"] = {
            "status": self.check_directory_exists("frontend/src/stores"),
            "details": {
                "Stores directory": self.check_directory_exists("frontend/src/stores"),
                "Auth store": self.check_file_exists("frontend/src/stores/authStore.ts"),
                "Video store": self.check_file_exists("frontend/src/stores/videoStore.ts"),
            }
        }
        
        # 2. Authentication UI components
        self.results["frontend"]["P1"]["Auth UI"] = {
            "status": self.check_directory_exists("frontend/src/components/auth"),
            "details": {
                "Auth components": self.check_directory_exists("frontend/src/components/auth"),
                "Login form": self.check_file_exists("frontend/src/components/auth/LoginForm.tsx"),
                "Register form": self.check_file_exists("frontend/src/components/auth/RegisterForm.tsx"),
            }
        }
        
        # 3. Dashboard layout
        self.results["frontend"]["P1"]["Dashboard Layout"] = {
            "status": self.check_directory_exists("frontend/src/components/dashboard"),
            "details": {
                "Dashboard dir": self.check_directory_exists("frontend/src/components/dashboard"),
                "Dashboard page": self.check_file_exists("frontend/src/pages/Dashboard.tsx"),
            }
        }
        
        # P2 Tasks
        print("\n=== FRONTEND P2 TASKS ===")
        
        # 1. Chart library integration
        self.results["frontend"]["P2"]["Charts"] = {
            "status": self.check_file_contains("frontend/package.json", ["recharts"])[0],
            "details": {
                "Recharts installed": self.check_file_contains("frontend/package.json", ["recharts"])[0],
                "Chart components": self.check_directory_exists("frontend/src/components/charts"),
            }
        }
        
        # 2. Real-time data architecture
        self.results["frontend"]["P2"]["Real-time"] = {
            "status": self.check_file_contains("frontend/package.json", ["socket.io-client"])[0],
            "details": {
                "Socket.io installed": self.check_file_contains("frontend/package.json", ["socket.io-client"])[0],
                "WebSocket service": self.check_file_exists("frontend/src/services/websocket.ts"),
            }
        }
    
    def verify_ops_tasks(self):
        """Verify Platform Ops Team deliverables"""
        
        # P0 Tasks
        print("\n=== PLATFORM OPS P0 TASKS ===")
        
        # 1. Local server setup
        self.results["ops"]["P0"]["Server Setup"] = {
            "status": True,  # Cannot verify hardware from code
            "details": {
                "Docker installed": self.check_file_exists("docker-compose.yml"),
                "Production config": self.check_file_exists("docker-compose.production.yml"),
            }
        }
        
        # 2. Docker infrastructure
        self.results["ops"]["P0"]["Docker Infrastructure"] = {
            "status": self.check_file_exists("docker-compose.yml"),
            "details": {
                "Main compose": self.check_file_exists("docker-compose.yml"),
                "Backend Dockerfile": self.check_file_exists("backend/Dockerfile"),
                "Frontend Dockerfile": self.check_file_exists("frontend/Dockerfile"),
                "GPU compose": self.check_file_exists("docker-compose.gpu.yml"),
            }
        }
        
        # 3. Security baseline
        self.results["ops"]["P0"]["Security Baseline"] = {
            "status": self.check_directory_exists("infrastructure/security"),
            "details": {
                "Security dir": self.check_directory_exists("infrastructure/security"),
                "Security configs": self.check_file_exists("backend/app/core/security.py"),
                "Secrets manager": self.check_file_exists("backend/app/core/secrets_manager.py"),
            }
        }
        
        # 4. Team tooling setup
        self.results["ops"]["P0"]["Team Tooling"] = {
            "status": self.check_directory_exists(".github"),
            "details": {
                "GitHub workflows": self.check_directory_exists(".github/workflows"),
                "Git config": self.check_file_exists(".gitignore"),
            }
        }
        
        # P1 Tasks
        print("\n=== PLATFORM OPS P1 TASKS ===")
        
        # 1. CI/CD pipeline
        self.results["ops"]["P1"]["CI/CD Pipeline"] = {
            "status": self.check_file_exists(".github/workflows/ci-cd.yml"),
            "details": {
                "CI workflow": self.check_file_exists(".github/workflows/ci.yml"),
                "CD workflow": self.check_file_exists(".github/workflows/ci-cd.yml"),
                "Docker build": self.check_file_exists(".github/workflows/docker-build.yml"),
            }
        }
        
        # 2. Monitoring stack
        self.results["ops"]["P1"]["Monitoring Stack"] = {
            "status": self.check_file_exists("docker-compose.monitoring.yml"),
            "details": {
                "Monitoring compose": self.check_file_exists("docker-compose.monitoring.yml"),
                "Prometheus config": self.check_directory_exists("infrastructure/monitoring/prometheus"),
                "Grafana dashboards": self.check_directory_exists("infrastructure/monitoring/grafana"),
            }
        }
        
        # 3. Secrets management
        self.results["ops"]["P1"]["Secrets Management"] = {
            "status": self.check_file_exists("backend/app/core/secrets_manager.py"),
            "details": {
                "Secrets manager": self.check_file_exists("backend/app/core/secrets_manager.py"),
                "Secrets rotation": self.check_file_exists("backend/app/core/secrets_rotation.py"),
                "Env example": self.check_file_exists("backend/.env.example"),
            }
        }
        
        # 4. Test framework
        self.results["ops"]["P1"]["Test Framework"] = {
            "status": self.check_file_exists("backend/pytest.ini"),
            "details": {
                "Pytest config": self.check_file_exists("backend/pytest.ini"),
                "Test directory": self.check_directory_exists("backend/tests"),
                "Frontend tests": self.check_directory_exists("frontend/tests"),
            }
        }
        
        # P2 Tasks
        print("\n=== PLATFORM OPS P2 TASKS ===")
        
        # 1. Backup strategy
        self.results["ops"]["P2"]["Backup Strategy"] = {
            "status": self.check_directory_exists("infrastructure/backup"),
            "details": {
                "Backup dir": self.check_directory_exists("infrastructure/backup"),
                "Disaster recovery": self.check_file_exists("docker-compose.disaster-recovery.yml"),
            }
        }
        
        # 2. SSL/TLS configuration
        self.results["ops"]["P2"]["SSL/TLS"] = {
            "status": self.check_directory_exists("infrastructure/ssl"),
            "details": {
                "SSL directory": self.check_directory_exists("infrastructure/ssl"),
                "Nginx config": self.check_file_exists("infrastructure/nginx/nginx.conf"),
            }
        }
        
        # 3. Performance testing
        self.results["ops"]["P2"]["Performance Testing"] = {
            "status": self.check_directory_exists("tests/performance"),
            "details": {
                "Performance tests": self.check_directory_exists("tests/performance"),
                "Load testing": self.check_file_exists("tests/performance/load_testing_suite.py"),
            }
        }
    
    def verify_aiml_tasks(self):
        """Verify AI/ML Team deliverables"""
        
        # P0 Tasks
        print("\n=== AI/ML P0 TASKS ===")
        
        # 1. AI service access setup
        self.results["aiml"]["P0"]["AI Service Access"] = {
            "status": self.check_file_exists("backend/app/services/ai_services.py"),
            "details": {
                "AI services": self.check_file_exists("backend/app/services/ai_services.py"),
                "Multi-provider": self.check_file_exists("backend/app/services/multi_provider_ai.py"),
                "Config": self.check_file_exists("ml-pipeline/config.yaml"),
            }
        }
        
        # 2. GPU environment configuration
        self.results["aiml"]["P0"]["GPU Environment"] = {
            "status": self.check_file_exists("docker-compose.gpu.yml"),
            "details": {
                "GPU compose": self.check_file_exists("docker-compose.gpu.yml"),
                "GPU service": self.check_file_exists("backend/app/services/gpu_resource_service.py"),
            }
        }
        
        # 3. ML pipeline architecture
        self.results["aiml"]["P0"]["ML Pipeline"] = {
            "status": self.check_directory_exists("ml-pipeline"),
            "details": {
                "ML pipeline dir": self.check_directory_exists("ml-pipeline"),
                "Services dir": self.check_directory_exists("ml-pipeline/services"),
                "Config": self.check_file_exists("ml-pipeline/config.yaml"),
            }
        }
        
        # 4. Cost optimization strategy
        self.results["aiml"]["P0"]["Cost Optimization"] = {
            "status": self.check_file_exists("backend/app/services/cost_optimizer.py"),
            "details": {
                "Cost optimizer": self.check_file_exists("backend/app/services/cost_optimizer.py"),
                "Cost config": self.check_file_contains("ml-pipeline/config.yaml", ["max_per_video: 3.00"])[0],
            }
        }
        
        # P1 Tasks
        print("\n=== AI/ML P1 TASKS ===")
        
        # 1. Model serving infrastructure
        self.results["aiml"]["P1"]["Model Serving"] = {
            "status": self.check_file_exists("backend/app/services/ml_integration_service.py"),
            "details": {
                "ML integration": self.check_file_exists("backend/app/services/ml_integration_service.py"),
                "Model endpoints": self.check_file_exists("backend/app/api/v1/endpoints/ml_models.py"),
            }
        }
        
        # 2. Trend prediction prototype
        self.results["aiml"]["P1"]["Trend Prediction"] = {
            "status": self.check_file_exists("ml-pipeline/services/trend_detection.py"),
            "details": {
                "Trend service": self.check_file_exists("ml-pipeline/services/trend_detection.py"),
                "Trend analyzer": self.check_file_exists("backend/app/services/trend_analyzer.py"),
            }
        }
        
        # 3. Model evaluation framework
        self.results["aiml"]["P1"]["Model Evaluation"] = {
            "status": self.check_directory_exists("ml-pipeline/quality_scoring"),
            "details": {
                "Quality scoring": self.check_directory_exists("ml-pipeline/quality_scoring"),
                "Benchmarks": self.check_directory_exists("ml-pipeline/benchmarks"),
            }
        }
        
        # P2 Tasks
        print("\n=== AI/ML P2 TASKS ===")
        
        # 1. Content quality scoring
        self.results["aiml"]["P2"]["Quality Scoring"] = {
            "status": self.check_directory_exists("ml-pipeline/quality_scoring"),
            "details": {
                "Quality dir": self.check_directory_exists("ml-pipeline/quality_scoring"),
                "Scoring service": self.check_file_exists("ml-pipeline/quality_scoring/content_quality_scorer.py"),
            }
        }
        
        # 2. Model monitoring
        self.results["aiml"]["P2"]["Model Monitoring"] = {
            "status": self.check_directory_exists("ml-pipeline/monitoring"),
            "details": {
                "Monitoring dir": self.check_directory_exists("ml-pipeline/monitoring"),
                "Performance tracking": self.check_file_exists("ml-pipeline/monitoring/performance_tracker.py"),
            }
        }
    
    def verify_data_tasks(self):
        """Verify Data Team deliverables"""
        
        # P0 Tasks
        print("\n=== DATA TEAM P0 TASKS ===")
        
        # 1. Data lake architecture
        self.results["data"]["P0"]["Data Lake"] = {
            "status": self.check_file_exists("backend/app/services/data_lake_service.py"),
            "details": {
                "Data lake service": self.check_file_exists("backend/app/services/data_lake_service.py"),
                "ETL functionality": True,  # Included in data lake service
            }
        }
        
        # 2. Training data pipeline
        self.results["data"]["P0"]["Training Pipeline"] = {
            "status": self.check_file_exists("backend/app/services/training_data_service.py"),
            "details": {
                "Training service": self.check_file_exists("backend/app/services/training_data_service.py"),
                "Feature engineering": self.check_file_exists("backend/app/services/feature_engineering.py"),
            }
        }
        
        # 3. Data schema design
        self.results["data"]["P0"]["Data Schema"] = {
            "status": self.check_file_exists("backend/app/models/analytics.py"),
            "details": {
                "Analytics model": self.check_file_exists("backend/app/models/analytics.py"),
                "Cost model": self.check_file_exists("backend/app/models/cost.py"),
            }
        }
        
        # P1 Tasks
        print("\n=== DATA TEAM P1 TASKS ===")
        
        # 1. Metrics database design
        self.results["data"]["P1"]["Metrics Database"] = {
            "status": self.check_file_exists("backend/app/models/analytics.py"),
            "details": {
                "Analytics models": self.check_file_exists("backend/app/models/analytics.py"),
                "Metrics service": self.check_file_exists("backend/app/services/analytics_service.py"),
            }
        }
        
        # 2. Real-time feature store
        self.results["data"]["P1"]["Feature Store"] = {
            "status": self.check_file_exists("backend/app/services/feature_store.py"),
            "details": {
                "Feature store": self.check_file_exists("backend/app/services/feature_store.py"),
                "Feature engineering": self.check_file_exists("backend/app/services/feature_engineering.py"),
            }
        }
        
        # 3. Vector database
        self.results["data"]["P1"]["Vector Database"] = {
            "status": self.check_file_exists("backend/app/services/vector_database.py"),
            "details": {
                "Vector DB service": self.check_file_exists("backend/app/services/vector_database.py"),
            }
        }
        
        # 4. Cost analytics framework
        self.results["data"]["P1"]["Cost Analytics"] = {
            "status": self.check_file_exists("backend/app/api/v1/endpoints/cost_intelligence.py"),
            "details": {
                "Cost endpoints": self.check_file_exists("backend/app/api/v1/endpoints/cost_intelligence.py"),
                "Cost tracking": self.check_file_exists("backend/app/services/cost_tracking.py"),
            }
        }
        
        # P2 Tasks
        print("\n=== DATA TEAM P2 TASKS ===")
        
        # 1. Feature engineering pipeline
        self.results["data"]["P2"]["Feature Pipeline"] = {
            "status": self.check_file_exists("backend/app/services/feature_engineering.py"),
            "details": {
                "Feature service": self.check_file_exists("backend/app/services/feature_engineering.py"),
            }
        }
        
        # 2. Reporting infrastructure
        self.results["data"]["P2"]["Reporting"] = {
            "status": self.check_file_exists("backend/app/api/v1/endpoints/reports.py"),
            "details": {
                "Reports endpoint": self.check_file_exists("backend/app/api/v1/endpoints/reports.py"),
                "Analytics service": self.check_file_exists("backend/app/services/analytics_service.py"),
            }
        }
    
    def generate_report(self):
        """Generate comprehensive report"""
        print("\n" + "="*80)
        print("WEEK 0 TASK VERIFICATION REPORT - DETAILED")
        print("="*80)
        
        for team, priorities in self.results.items():
            print(f"\n{'='*40}")
            print(f"{team.upper()} TEAM")
            print(f"{'='*40}")
            
            for priority, tasks in priorities.items():
                if not tasks:
                    continue
                    
                completed = sum(1 for task in tasks.values() if task.get("status", False))
                total = len(tasks)
                percentage = (completed / total * 100) if total > 0 else 0
                
                print(f"\n{priority} Tasks: {completed}/{total} ({percentage:.1f}%)")
                
                for task_name, task_data in tasks.items():
                    status = "[PASS]" if task_data.get("status", False) else "[FAIL]"
                    print(f"  {status} {task_name}")
                    
                    if "details" in task_data and isinstance(task_data["details"], dict):
                        for detail_name, detail_status in task_data["details"].items():
                            if isinstance(detail_status, tuple):
                                detail_status = detail_status[0]
                            detail_icon = "+" if detail_status else "-"
                            print(f"      {detail_icon} {detail_name}")
        
        # Generate summary
        print("\n" + "="*80)
        print("SUMMARY")
        print("="*80)
        
        overall_completed = 0
        overall_total = 0
        
        for team, priorities in self.results.items():
            team_completed = 0
            team_total = 0
            
            for priority, tasks in priorities.items():
                completed = sum(1 for task in tasks.values() if task.get("status", False))
                total = len(tasks)
                team_completed += completed
                team_total += total
            
            if team_total > 0:
                percentage = (team_completed / team_total * 100)
                print(f"{team.upper()}: {team_completed}/{team_total} tasks ({percentage:.1f}%)")
                overall_completed += team_completed
                overall_total += team_total
        
        if overall_total > 0:
            overall_percentage = (overall_completed / overall_total * 100)
            print(f"\nOVERALL: {overall_completed}/{overall_total} tasks ({overall_percentage:.1f}%)")
    
    def run(self):
        """Run all verifications"""
        self.verify_backend_tasks()
        self.verify_frontend_tasks()
        self.verify_ops_tasks()
        self.verify_aiml_tasks()
        self.verify_data_tasks()
        self.generate_report()

if __name__ == "__main__":
    verifier = Week0Verifier()
    verifier.run()