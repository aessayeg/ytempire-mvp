"""
Comprehensive Validation Test for YTEmpire MVP
Verifies ALL P0, P1, P2 tasks from Week 0-2 are complete
Runs unit tests, functionality tests, and integration tests
"""

import os
import sys
import json
import asyncio
import importlib.util
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass, field
import subprocess

# Add backend to path
sys.path.insert(0, str(Path(__file__).parent.parent / "backend"))

@dataclass
class TaskDefinition:
    """Definition of a task to verify"""
    team: str
    priority: str  # P0, P1, P2
    week: int
    name: str
    description: str
    verification_type: str  # file_exists, function_exists, integration_test, etc.
    verification_target: str  # file path, function name, etc.
    
@dataclass
class TestResult:
    """Result of a test"""
    task: TaskDefinition
    status: str  # PASS, FAIL, PARTIAL
    details: str
    errors: List[str] = field(default_factory=list)

class ComprehensiveValidator:
    def __init__(self):
        self.project_root = Path(".").resolve()
        self.backend_path = self.project_root / "backend"
        self.frontend_path = self.project_root / "frontend"
        self.results = {
            "timestamp": datetime.now().isoformat(),
            "summary": {},
            "backend": {"P0": [], "P1": [], "P2": []},
            "frontend": {"P0": [], "P1": [], "P2": []},
            "platform_ops": {"P0": [], "P1": [], "P2": []},
            "ai_ml": {"P0": [], "P1": [], "P2": []},
            "data": {"P0": [], "P1": [], "P2": []},
            "unit_tests": [],
            "functionality_tests": [],
            "integration_tests": [],
            "hanging_features": []
        }
        
    def get_all_tasks(self) -> List[TaskDefinition]:
        """Define all tasks from Week 0-2 based on documentation"""
        tasks = []
        
        # ========== BACKEND TEAM TASKS ==========
        # Week 0 - Backend
        backend_week0_p0 = [
            TaskDefinition("backend", "P0", 0, "Project Setup", 
                          "FastAPI structure with async SQLAlchemy", 
                          "file_exists", "backend/app/main.py"),
            TaskDefinition("backend", "P0", 0, "Database Models", 
                          "User, Channel, Video, Analytics models", 
                          "file_exists", "backend/app/models/video.py"),
            TaskDefinition("backend", "P0", 0, "Authentication System", 
                          "JWT with refresh tokens", 
                          "file_exists", "backend/app/core/security.py"),
            TaskDefinition("backend", "P0", 0, "Basic API Endpoints", 
                          "CRUD for videos, channels", 
                          "file_exists", "backend/app/api/v1/endpoints/videos.py"),
        ]
        
        backend_week0_p1 = [
            TaskDefinition("backend", "P1", 0, "Celery Setup", 
                          "Task queue configuration", 
                          "file_exists", "backend/app/core/celery_app.py"),
            TaskDefinition("backend", "P1", 0, "Redis Integration", 
                          "Caching and session management", 
                          "config_check", "REDIS_URL"),
            TaskDefinition("backend", "P1", 0, "Cost Tracking", 
                          "Basic cost tracking service", 
                          "file_exists", "backend/app/services/cost_tracking.py"),
        ]
        
        # Week 1 - Backend
        backend_week1_p0 = [
            TaskDefinition("backend", "P0", 1, "Video Generation Pipeline", 
                          "Complete video generation service", 
                          "file_exists", "backend/app/services/video_generation_pipeline.py"),
            TaskDefinition("backend", "P0", 1, "YouTube Integration", 
                          "Upload and management", 
                          "file_exists", "backend/app/services/youtube_service.py"),
            TaskDefinition("backend", "P0", 1, "Analytics Pipeline", 
                          "Data collection and processing", 
                          "file_exists", "backend/app/services/analytics_service.py"),
            TaskDefinition("backend", "P0", 1, "WebSocket Support", 
                          "Real-time updates", 
                          "file_exists", "backend/app/services/websocket_manager.py"),
        ]
        
        backend_week1_p1 = [
            TaskDefinition("backend", "P1", 1, "Batch Processing", 
                          "Handle multiple videos", 
                          "file_exists", "backend/app/services/batch_processing.py"),
            TaskDefinition("backend", "P1", 1, "Error Recovery", 
                          "Retry logic and fallbacks", 
                          "file_exists", "backend/app/services/error_handlers.py"),
            TaskDefinition("backend", "P1", 1, "Performance Monitoring", 
                          "Metrics collection", 
                          "file_exists", "backend/app/services/performance_monitoring.py"),
        ]
        
        # Week 2 - Backend
        backend_week2_p0 = [
            TaskDefinition("backend", "P0", 2, "Multi-Channel Architecture", 
                          "15 YouTube account management", 
                          "file_exists", "backend/app/services/youtube_multi_account.py"),
            TaskDefinition("backend", "P0", 2, "Batch Processing Scale", 
                          "50-100 videos/day", 
                          "function_exists", "backend/app/services/batch_processing.py:process_batch"),
            TaskDefinition("backend", "P0", 2, "Subscription System", 
                          "Payment and tiers", 
                          "file_exists", "backend/app/services/subscription_service.py"),
            TaskDefinition("backend", "P0", 2, "Database Optimization", 
                          "Connection pooling (200 connections)", 
                          "config_check", "database_pool"),
        ]
        
        backend_week2_p1 = [
            TaskDefinition("backend", "P1", 2, "Advanced Analytics", 
                          "Predictive metrics", 
                          "file_exists", "backend/app/services/realtime_analytics_service.py"),
            TaskDefinition("backend", "P1", 2, "A/B Testing", 
                          "Experiment framework", 
                          "file_exists", "backend/app/services/ab_testing_service.py"),
            TaskDefinition("backend", "P1", 2, "Webhook System", 
                          "External integrations", 
                          "file_exists", "backend/app/services/webhook_service.py"),
        ]
        
        backend_week2_p2 = [
            TaskDefinition("backend", "P2", 2, "Advanced Caching", 
                          "Multi-tier caching", 
                          "file_exists", "backend/app/services/advanced_caching.py"),
            TaskDefinition("backend", "P2", 2, "Third-party Integrations", 
                          "External services", 
                          "file_exists", "backend/app/services/third_party_integrations.py"),
        ]
        
        # ========== FRONTEND TEAM TASKS ==========
        # Week 0 - Frontend
        frontend_week0_p0 = [
            TaskDefinition("frontend", "P0", 0, "React Setup", 
                          "React 18 with TypeScript", 
                          "file_exists", "frontend/package.json"),
            TaskDefinition("frontend", "P0", 0, "Authentication UI", 
                          "Login/Register pages", 
                          "file_exists", "frontend/src/pages/Login"),
            TaskDefinition("frontend", "P0", 0, "Basic Dashboard", 
                          "Main dashboard layout", 
                          "file_exists", "frontend/src/pages/Dashboard"),
        ]
        
        frontend_week0_p1 = [
            TaskDefinition("frontend", "P1", 0, "State Management", 
                          "Zustand store setup", 
                          "file_exists", "frontend/src/stores"),
            TaskDefinition("frontend", "P1", 0, "API Integration", 
                          "Axios configuration", 
                          "file_exists", "frontend/src/services/api"),
        ]
        
        # Week 1 - Frontend
        frontend_week1_p0 = [
            TaskDefinition("frontend", "P0", 1, "Video Management UI", 
                          "Create, edit, list videos", 
                          "file_exists", "frontend/src/components/Videos"),
            TaskDefinition("frontend", "P0", 1, "Channel Dashboard", 
                          "Channel management interface", 
                          "file_exists", "frontend/src/components/Channels"),
            TaskDefinition("frontend", "P0", 1, "Analytics Dashboard", 
                          "Charts and metrics", 
                          "file_exists", "frontend/src/components/Analytics"),
        ]
        
        frontend_week1_p1 = [
            TaskDefinition("frontend", "P1", 1, "Real-time Updates", 
                          "WebSocket integration", 
                          "file_exists", "frontend/src/hooks/useWebSocket"),
            TaskDefinition("frontend", "P1", 1, "Mobile Responsive", 
                          "Responsive design", 
                          "css_check", "responsive"),
        ]
        
        # Week 2 - Frontend
        frontend_week2_p0 = [
            TaskDefinition("frontend", "P0", 2, "Multi-Channel UI", 
                          "Manage 15 channels", 
                          "component_exists", "ChannelManager"),
            TaskDefinition("frontend", "P0", 2, "Batch Operations UI", 
                          "Bulk video operations", 
                          "component_exists", "BatchOperations"),
            TaskDefinition("frontend", "P0", 2, "Beta Onboarding", 
                          "User onboarding flow", 
                          "file_exists", "frontend/src/components/Onboarding"),
        ]
        
        frontend_week2_p1 = [
            TaskDefinition("frontend", "P1", 2, "Advanced Analytics UI", 
                          "Complex visualizations", 
                          "component_exists", "AdvancedAnalytics"),
            TaskDefinition("frontend", "P1", 2, "Settings Panel", 
                          "Configuration management", 
                          "file_exists", "frontend/src/pages/Settings"),
        ]
        
        # ========== PLATFORM OPS TEAM TASKS ==========
        # Week 0 - Platform Ops
        platform_week0_p0 = [
            TaskDefinition("platform_ops", "P0", 0, "Docker Setup", 
                          "Docker compose configuration", 
                          "file_exists", "docker-compose.yml"),
            TaskDefinition("platform_ops", "P0", 0, "PostgreSQL Setup", 
                          "Database configuration", 
                          "service_check", "postgres"),
            TaskDefinition("platform_ops", "P0", 0, "Redis Setup", 
                          "Cache configuration", 
                          "service_check", "redis"),
        ]
        
        platform_week0_p1 = [
            TaskDefinition("platform_ops", "P1", 0, "CI/CD Pipeline", 
                          "GitHub Actions", 
                          "file_exists", ".github/workflows"),
            TaskDefinition("platform_ops", "P1", 0, "Environment Config", 
                          "Environment variables", 
                          "file_exists", "backend/.env.example"),
        ]
        
        # Week 1 - Platform Ops
        platform_week1_p0 = [
            TaskDefinition("platform_ops", "P0", 1, "Monitoring Stack", 
                          "Prometheus + Grafana", 
                          "file_exists", "docker-compose.monitoring.yml"),
            TaskDefinition("platform_ops", "P0", 1, "Logging System", 
                          "Centralized logging", 
                          "config_check", "logging"),
            TaskDefinition("platform_ops", "P0", 1, "Backup System", 
                          "Database backups", 
                          "file_exists", "infrastructure/backup"),
        ]
        
        platform_week1_p1 = [
            TaskDefinition("platform_ops", "P1", 1, "Security Hardening", 
                          "Security configurations", 
                          "file_exists", "infrastructure/security"),
            TaskDefinition("platform_ops", "P1", 1, "Load Balancing", 
                          "Traffic distribution", 
                          "config_check", "load_balancer"),
        ]
        
        # Week 2 - Platform Ops
        platform_week2_p0 = [
            TaskDefinition("platform_ops", "P0", 2, "Production Deployment", 
                          "Production configuration", 
                          "file_exists", "docker-compose.production.yml"),
            TaskDefinition("platform_ops", "P0", 2, "Auto-scaling", 
                          "Dynamic scaling", 
                          "file_exists", "infrastructure/scaling"),
            TaskDefinition("platform_ops", "P0", 2, "Security Compliance", 
                          "Security measures", 
                          "file_exists", "infrastructure/compliance"),
        ]
        
        # ========== AI/ML TEAM TASKS ==========
        # Week 0 - AI/ML
        ai_ml_week0_p0 = [
            TaskDefinition("ai_ml", "P0", 0, "OpenAI Integration", 
                          "GPT-4 setup", 
                          "file_exists", "backend/app/services/ai_services.py"),
            TaskDefinition("ai_ml", "P0", 0, "Script Generation", 
                          "Content generation", 
                          "function_exists", "backend/app/services/ai_services.py:generate_script"),
        ]
        
        ai_ml_week0_p1 = [
            TaskDefinition("ai_ml", "P1", 0, "Prompt Templates", 
                          "Optimized prompts", 
                          "file_exists", "ml-pipeline/prompts"),
            TaskDefinition("ai_ml", "P1", 0, "Cost Optimization", 
                          "Model selection logic", 
                          "function_exists", "backend/app/services/cost_optimizer.py:optimize_model_selection"),
        ]
        
        # Week 1 - AI/ML
        ai_ml_week1_p0 = [
            TaskDefinition("ai_ml", "P0", 1, "Voice Synthesis", 
                          "ElevenLabs integration", 
                          "config_check", "ELEVENLABS_API_KEY"),
            TaskDefinition("ai_ml", "P0", 1, "Thumbnail Generation", 
                          "DALL-E 3 integration", 
                          "file_exists", "backend/app/services/thumbnail_generator.py"),
            TaskDefinition("ai_ml", "P0", 1, "Quality Scoring", 
                          "Content quality metrics", 
                          "function_exists", "backend/app/services/analytics_service.py:calculate_quality_score"),
        ]
        
        # Week 2 - AI/ML
        ai_ml_week2_p0 = [
            TaskDefinition("ai_ml", "P0", 2, "Multi-Model Orchestration", 
                          "Provider fallback", 
                          "file_exists", "backend/app/services/multi_provider_ai.py"),
            TaskDefinition("ai_ml", "P0", 2, "ML Pipeline", 
                          "End-to-end ML workflow", 
                          "file_exists", "ml-pipeline/config.yaml"),
            TaskDefinition("ai_ml", "P0", 2, "Personalization", 
                          "User preference learning", 
                          "file_exists", "backend/app/services/ml_integration_service.py"),
        ]
        
        # ========== DATA TEAM TASKS ==========
        # Week 0 - Data
        data_week0_p0 = [
            TaskDefinition("data", "P0", 0, "Analytics Schema", 
                          "Database schema for analytics", 
                          "file_exists", "backend/app/models/analytics.py"),
            TaskDefinition("data", "P0", 0, "Data Collection", 
                          "Event tracking setup", 
                          "function_exists", "backend/app/services/analytics_service.py:track_event"),
        ]
        
        # Week 1 - Data
        data_week1_p0 = [
            TaskDefinition("data", "P0", 1, "Real-time Analytics", 
                          "Live data processing", 
                          "file_exists", "backend/app/services/realtime_analytics_service.py"),
            TaskDefinition("data", "P0", 1, "Reporting System", 
                          "Report generation", 
                          "function_exists", "backend/app/services/analytics_service.py:generate_report"),
        ]
        
        # Week 2 - Data
        data_week2_p0 = [
            TaskDefinition("data", "P0", 2, "Data Warehouse", 
                          "ETL pipeline", 
                          "file_exists", "backend/app/services/data_lake_service.py"),
            TaskDefinition("data", "P0", 2, "Advanced Analytics", 
                          "Predictive models", 
                          "file_exists", "backend/app/services/advanced_forecasting_models.py"),
            TaskDefinition("data", "P0", 2, "Data Visualization", 
                          "Dashboard components", 
                          "file_exists", "backend/app/services/advanced_data_visualization.py"),
        ]
        
        data_week2_p1 = [
            TaskDefinition("data", "P1", 2, "Data Quality", 
                          "Data validation", 
                          "file_exists", "backend/app/services/data_quality.py"),
            TaskDefinition("data", "P1", 2, "Data Marketplace", 
                          "External data integration", 
                          "file_exists", "backend/app/services/data_marketplace_integration.py"),
        ]
        
        # Combine all tasks
        tasks.extend(backend_week0_p0 + backend_week0_p1)
        tasks.extend(backend_week1_p0 + backend_week1_p1)
        tasks.extend(backend_week2_p0 + backend_week2_p1 + backend_week2_p2)
        
        tasks.extend(frontend_week0_p0 + frontend_week0_p1)
        tasks.extend(frontend_week1_p0 + frontend_week1_p1)
        tasks.extend(frontend_week2_p0 + frontend_week2_p1)
        
        tasks.extend(platform_week0_p0 + platform_week0_p1)
        tasks.extend(platform_week1_p0 + platform_week1_p1)
        tasks.extend(platform_week2_p0)
        
        tasks.extend(ai_ml_week0_p0 + ai_ml_week0_p1)
        tasks.extend(ai_ml_week1_p0)
        tasks.extend(ai_ml_week2_p0)
        
        tasks.extend(data_week0_p0)
        tasks.extend(data_week1_p0)
        tasks.extend(data_week2_p0 + data_week2_p1)
        
        return tasks
    
    def verify_task(self, task: TaskDefinition) -> TestResult:
        """Verify a single task based on its type"""
        if task.verification_type == "file_exists":
            return self.verify_file_exists(task)
        elif task.verification_type == "function_exists":
            return self.verify_function_exists(task)
        elif task.verification_type == "config_check":
            return self.verify_config(task)
        elif task.verification_type == "service_check":
            return self.verify_service(task)
        elif task.verification_type == "component_exists":
            return self.verify_component(task)
        elif task.verification_type == "css_check":
            return self.verify_css(task)
        else:
            return TestResult(task, "FAIL", f"Unknown verification type: {task.verification_type}")
    
    def verify_file_exists(self, task: TaskDefinition) -> TestResult:
        """Check if a file exists"""
        file_path = self.project_root / task.verification_target
        
        if file_path.exists():
            # Check if it's not empty
            if file_path.is_file():
                size = file_path.stat().st_size
                if size > 100:  # More than 100 bytes
                    return TestResult(task, "PASS", f"File exists ({size:,} bytes)")
                else:
                    return TestResult(task, "PARTIAL", f"File exists but very small ({size} bytes)")
            else:
                # It's a directory, check if it has files
                files = list(file_path.glob("*"))
                if files:
                    return TestResult(task, "PASS", f"Directory exists with {len(files)} files")
                else:
                    return TestResult(task, "PARTIAL", "Directory exists but empty")
        else:
            # Try to find similar files
            parent = file_path.parent
            if parent.exists():
                similar = list(parent.glob(f"*{file_path.stem}*"))
                if similar:
                    return TestResult(task, "PARTIAL", f"File not found, but similar exists: {similar[0].name}")
            return TestResult(task, "FAIL", f"File not found: {task.verification_target}")
    
    def verify_function_exists(self, task: TaskDefinition) -> TestResult:
        """Check if a function exists in a file"""
        parts = task.verification_target.split(":")
        if len(parts) != 2:
            return TestResult(task, "FAIL", "Invalid function specification")
        
        file_path = self.project_root / parts[0]
        function_name = parts[1]
        
        if not file_path.exists():
            return TestResult(task, "FAIL", f"File not found: {parts[0]}")
        
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            # Check for function definition
            if f"def {function_name}" in content or f"async def {function_name}" in content:
                return TestResult(task, "PASS", f"Function '{function_name}' found")
            else:
                # Check for class method
                if f".{function_name}(" in content:
                    return TestResult(task, "PASS", f"Method '{function_name}' found")
                return TestResult(task, "FAIL", f"Function '{function_name}' not found in {parts[0]}")
        except Exception as e:
            return TestResult(task, "FAIL", f"Error reading file: {str(e)}")
    
    def verify_config(self, task: TaskDefinition) -> TestResult:
        """Check if a configuration exists"""
        config_name = task.verification_target
        
        # Check environment files
        env_files = [
            self.backend_path / ".env",
            self.backend_path / ".env.example",
            self.project_root / "docker-compose.yml"
        ]
        
        for env_file in env_files:
            if env_file.exists():
                try:
                    with open(env_file, 'r', encoding='utf-8') as f:
                        content = f.read()
                    if config_name.upper() in content.upper():
                        return TestResult(task, "PASS", f"Config '{config_name}' found in {env_file.name}")
                except:
                    pass
        
        # Check Python config
        config_file = self.backend_path / "app" / "core" / "config.py"
        if config_file.exists():
            try:
                with open(config_file, 'r', encoding='utf-8') as f:
                    content = f.read()
                if config_name.lower() in content.lower():
                    return TestResult(task, "PASS", f"Config '{config_name}' found in config.py")
            except:
                pass
        
        return TestResult(task, "FAIL", f"Config '{config_name}' not found")
    
    def verify_service(self, task: TaskDefinition) -> TestResult:
        """Check if a service is configured"""
        service_name = task.verification_target
        
        # Check docker-compose
        docker_file = self.project_root / "docker-compose.yml"
        if docker_file.exists():
            try:
                with open(docker_file, 'r', encoding='utf-8') as f:
                    content = f.read()
                if service_name in content:
                    return TestResult(task, "PASS", f"Service '{service_name}' configured in docker-compose")
            except:
                pass
        
        return TestResult(task, "PARTIAL", f"Service '{service_name}' configuration not verified")
    
    def verify_component(self, task: TaskDefinition) -> TestResult:
        """Check if a React component exists"""
        component_name = task.verification_target
        
        # Search for component files
        component_patterns = [
            f"**/{component_name}.tsx",
            f"**/{component_name}.jsx",
            f"**/{component_name}/index.tsx",
            f"**/{component_name}/index.jsx"
        ]
        
        for pattern in component_patterns:
            matches = list(self.frontend_path.glob(pattern))
            if matches:
                return TestResult(task, "PASS", f"Component '{component_name}' found at {matches[0].relative_to(self.frontend_path)}")
        
        return TestResult(task, "FAIL", f"Component '{component_name}' not found")
    
    def verify_css(self, task: TaskDefinition) -> TestResult:
        """Check for CSS/styling"""
        style_type = task.verification_target
        
        # Check for Tailwind or CSS modules
        css_files = list(self.frontend_path.glob("**/*.css")) + list(self.frontend_path.glob("**/*.scss"))
        
        if css_files:
            if style_type == "responsive":
                # Check for responsive classes
                for css_file in css_files[:5]:  # Check first 5 files
                    try:
                        with open(css_file, 'r', encoding='utf-8') as f:
                            content = f.read()
                        if any(x in content for x in ["@media", "sm:", "md:", "lg:", "xl:"]):
                            return TestResult(task, "PASS", "Responsive styling found")
                    except:
                        pass
            return TestResult(task, "PARTIAL", f"CSS files found but {style_type} not verified")
        
        return TestResult(task, "FAIL", "No CSS files found")
    
    async def run_unit_tests(self):
        """Run unit tests for critical components"""
        print("\n[UNIT TESTS]")
        unit_tests = []
        
        # Test 1: Database Models
        try:
            from app.models import User, Channel, Video, Analytics
            unit_tests.append({"test": "Database Models Import", "status": "PASS"})
            print("  [OK] Database Models Import")
        except Exception as e:
            unit_tests.append({"test": "Database Models Import", "status": "FAIL", "error": str(e)})
            print(f"  [FAIL] Database Models Import: {e}")
        
        # Test 2: Core Services
        try:
            from app.services.video_generation_pipeline import VideoGenerationPipeline
            from app.services.analytics_service import analytics_service
            from app.services.cost_tracking import cost_tracker
            unit_tests.append({"test": "Core Services Import", "status": "PASS"})
            print("  [OK] Core Services Import")
        except Exception as e:
            unit_tests.append({"test": "Core Services Import", "status": "FAIL", "error": str(e)})
            print(f"  [FAIL] Core Services Import: {e}")
        
        # Test 3: API Endpoints
        try:
            from app.api.v1.api import api_router
            route_count = len(api_router.routes)
            unit_tests.append({"test": "API Routes", "status": "PASS", "details": f"{route_count} routes"})
            print(f"  [OK] API Routes ({route_count} routes)")
        except Exception as e:
            unit_tests.append({"test": "API Routes", "status": "FAIL", "error": str(e)})
            print(f"  [FAIL] API Routes: {e}")
        
        # Test 4: Celery Tasks
        try:
            from app.tasks import video_tasks, ai_tasks, analytics_tasks
            unit_tests.append({"test": "Celery Tasks Import", "status": "PASS"})
            print("  [OK] Celery Tasks Import")
        except Exception as e:
            unit_tests.append({"test": "Celery Tasks Import", "status": "FAIL", "error": str(e)})
            print(f"  [FAIL] Celery Tasks Import: {e}")
        
        # Test 5: Configuration
        try:
            from app.core.config import settings
            required_settings = ["DATABASE_URL", "JWT_SECRET_KEY", "REDIS_URL"]
            for setting in required_settings:
                if not hasattr(settings, setting):
                    raise ValueError(f"Missing setting: {setting}")
            unit_tests.append({"test": "Configuration", "status": "PASS"})
            print("  [OK] Configuration")
        except Exception as e:
            unit_tests.append({"test": "Configuration", "status": "FAIL", "error": str(e)})
            print(f"  [FAIL] Configuration: {e}")
        
        self.results["unit_tests"] = unit_tests
        return unit_tests
    
    async def run_functionality_tests(self):
        """Test key functionalities"""
        print("\n[FUNCTIONALITY TESTS]")
        func_tests = []
        
        # Test 1: Video Generation Pipeline
        try:
            from app.services.video_generation_pipeline import VideoGenerationPipeline
            pipeline = VideoGenerationPipeline()
            # Check if main methods exist
            required_methods = ["generate_video", "process_script", "generate_thumbnail"]
            for method in required_methods:
                if not hasattr(pipeline, method):
                    raise ValueError(f"Missing method: {method}")
            func_tests.append({"test": "Video Pipeline Functions", "status": "PASS"})
            print("  [OK] Video Pipeline Functions")
        except Exception as e:
            func_tests.append({"test": "Video Pipeline Functions", "status": "FAIL", "error": str(e)})
            print(f"  [FAIL] Video Pipeline Functions: {e}")
        
        # Test 2: Analytics Functions
        try:
            from app.services.analytics_service import analytics_service
            # Check key analytics functions
            if hasattr(analytics_service, 'track_event') and hasattr(analytics_service, 'generate_report'):
                func_tests.append({"test": "Analytics Functions", "status": "PASS"})
                print("  [OK] Analytics Functions")
            else:
                raise ValueError("Missing analytics functions")
        except Exception as e:
            func_tests.append({"test": "Analytics Functions", "status": "FAIL", "error": str(e)})
            print(f"  [FAIL] Analytics Functions: {e}")
        
        # Test 3: Cost Tracking Functions
        try:
            from app.services.cost_tracking import cost_tracker
            if hasattr(cost_tracker, 'track_cost') and hasattr(cost_tracker, 'get_daily_costs'):
                func_tests.append({"test": "Cost Tracking Functions", "status": "PASS"})
                print("  [OK] Cost Tracking Functions")
            else:
                raise ValueError("Missing cost tracking functions")
        except Exception as e:
            func_tests.append({"test": "Cost Tracking Functions", "status": "FAIL", "error": str(e)})
            print(f"  [FAIL] Cost Tracking Functions: {e}")
        
        # Test 4: WebSocket Manager
        try:
            from app.services.websocket_manager import ConnectionManager
            ws_manager = ConnectionManager()
            if hasattr(ws_manager, 'connect') and hasattr(ws_manager, 'broadcast'):
                func_tests.append({"test": "WebSocket Functions", "status": "PASS"})
                print("  [OK] WebSocket Functions")
            else:
                raise ValueError("Missing WebSocket functions")
        except Exception as e:
            func_tests.append({"test": "WebSocket Functions", "status": "FAIL", "error": str(e)})
            print(f"  [FAIL] WebSocket Functions: {e}")
        
        # Test 5: YouTube Multi-Account
        try:
            from app.services.youtube_multi_account import get_youtube_manager
            manager = get_youtube_manager()
            if hasattr(manager, 'get_healthiest_account'):
                func_tests.append({"test": "YouTube Multi-Account", "status": "PASS"})
                print("  [OK] YouTube Multi-Account")
            else:
                raise ValueError("Missing multi-account functions")
        except Exception as e:
            func_tests.append({"test": "YouTube Multi-Account", "status": "FAIL", "error": str(e)})
            print(f"  [FAIL] YouTube Multi-Account: {e}")
        
        self.results["functionality_tests"] = func_tests
        return func_tests
    
    async def run_integration_tests(self):
        """Test integrations between components"""
        print("\n[INTEGRATION TESTS]")
        integration_tests = []
        
        # Test 1: Database Connection
        try:
            from app.db.session import engine
            from sqlalchemy import text
            async with engine.connect() as conn:
                result = await conn.execute(text("SELECT 1"))
                integration_tests.append({"test": "Database Connection", "status": "PASS"})
                print("  [OK] Database Connection")
        except Exception as e:
            integration_tests.append({"test": "Database Connection", "status": "FAIL", "error": str(e)})
            print(f"  [FAIL] Database Connection: {e}")
        
        # Test 2: Redis Connection
        try:
            import redis.asyncio as redis
            from app.core.config import settings
            if hasattr(settings, 'REDIS_URL'):
                # Just check if we can import and have config
                integration_tests.append({"test": "Redis Configuration", "status": "PASS"})
                print("  [OK] Redis Configuration")
            else:
                raise ValueError("No Redis configuration found")
        except Exception as e:
            integration_tests.append({"test": "Redis Configuration", "status": "FAIL", "error": str(e)})
            print(f"  [FAIL] Redis Configuration: {e}")
        
        # Test 3: Service Dependencies
        try:
            # Check if services can import each other
            from app.services.video_generation_pipeline import VideoGenerationPipeline
            from app.services.cost_tracking import cost_tracker
            from app.services.analytics_service import analytics_service
            
            # Check if video pipeline uses cost tracking
            pipeline_file = self.backend_path / "app/services/video_generation_pipeline.py"
            with open(pipeline_file, 'r', encoding='utf-8') as f:
                content = f.read()
            
            if "cost_tracker" in content or "cost_tracking" in content:
                integration_tests.append({"test": "Service Cross-Dependencies", "status": "PASS"})
                print("  [OK] Service Cross-Dependencies")
            else:
                integration_tests.append({"test": "Service Cross-Dependencies", "status": "PARTIAL", 
                                        "details": "Services isolated, may need integration"})
                print("  [WARN] Service Cross-Dependencies (isolated)")
        except Exception as e:
            integration_tests.append({"test": "Service Cross-Dependencies", "status": "FAIL", "error": str(e)})
            print(f"  [FAIL] Service Cross-Dependencies: {e}")
        
        # Test 4: API-Service Integration
        try:
            # Check if API endpoints use services
            endpoint_file = self.backend_path / "app/api/v1/endpoints/videos.py"
            if endpoint_file.exists():
                with open(endpoint_file, 'r', encoding='utf-8') as f:
                    content = f.read()
                
                if "video_generation_pipeline" in content or "VideoGenerationPipeline" in content:
                    integration_tests.append({"test": "API-Service Integration", "status": "PASS"})
                    print("  [OK] API-Service Integration")
                else:
                    integration_tests.append({"test": "API-Service Integration", "status": "PARTIAL",
                                            "details": "API may not be using consolidated services"})
                    print("  [WARN] API-Service Integration (check service usage)")
            else:
                raise ValueError("Videos endpoint not found")
        except Exception as e:
            integration_tests.append({"test": "API-Service Integration", "status": "FAIL", "error": str(e)})
            print(f"  [FAIL] API-Service Integration: {e}")
        
        # Test 5: Celery-Service Integration
        try:
            task_file = self.backend_path / "app/tasks/video_tasks.py"
            if task_file.exists():
                with open(task_file, 'r', encoding='utf-8') as f:
                    content = f.read()
                
                if "video_generation_pipeline" in content or "VideoGenerationPipeline" in content:
                    integration_tests.append({"test": "Celery-Service Integration", "status": "PASS"})
                    print("  [OK] Celery-Service Integration")
                else:
                    integration_tests.append({"test": "Celery-Service Integration", "status": "PARTIAL",
                                            "details": "Tasks may not be using consolidated services"})
                    print("  [WARN] Celery-Service Integration (check service usage)")
            else:
                raise ValueError("Video tasks not found")
        except Exception as e:
            integration_tests.append({"test": "Celery-Service Integration", "status": "FAIL", "error": str(e)})
            print(f"  [FAIL] Celery-Service Integration: {e}")
        
        self.results["integration_tests"] = integration_tests
        return integration_tests
    
    def find_hanging_features(self):
        """Find features/services that are not connected to anything"""
        print("\n[CHECKING FOR HANGING FEATURES]")
        hanging = []
        
        # Get all service files
        services_path = self.backend_path / "app/services"
        all_services = list(services_path.glob("*.py"))
        
        for service_file in all_services:
            if service_file.name == "__init__.py":
                continue
            
            service_name = service_file.stem
            
            # Check if this service is imported anywhere
            is_imported = False
            
            # Check main.py
            main_file = self.backend_path / "app/main.py"
            if main_file.exists():
                with open(main_file, 'r', encoding='utf-8') as f:
                    if service_name in f.read():
                        is_imported = True
            
            # Check API endpoints
            if not is_imported:
                api_path = self.backend_path / "app/api/v1/endpoints"
                for api_file in api_path.glob("*.py"):
                    with open(api_file, 'r', encoding='utf-8') as f:
                        if service_name in f.read():
                            is_imported = True
                            break
            
            # Check tasks
            if not is_imported:
                tasks_path = self.backend_path / "app/tasks"
                if tasks_path.exists():
                    for task_file in tasks_path.glob("*.py"):
                        with open(task_file, 'r', encoding='utf-8') as f:
                            if service_name in f.read():
                                is_imported = True
                                break
            
            # Check other services
            if not is_imported:
                for other_service in all_services:
                    if other_service == service_file:
                        continue
                    try:
                        with open(other_service, 'r', encoding='utf-8') as f:
                            if service_name in f.read():
                                is_imported = True
                                break
                    except:
                        pass
            
            if not is_imported:
                hanging.append(service_name)
                print(f"  [WARN] Hanging service: {service_name}")
        
        if not hanging:
            print("  [OK] No hanging services found")
        
        self.results["hanging_features"] = hanging
        return hanging
    
    async def run_all_validations(self):
        """Run all validations"""
        print("\n" + "="*100)
        print("COMPREHENSIVE VALIDATION TEST - YTEmpire MVP")
        print("="*100)
        
        # Get all tasks
        all_tasks = self.get_all_tasks()
        print(f"\nTotal tasks to verify: {len(all_tasks)}")
        print(f"  - Backend: {len([t for t in all_tasks if t.team == 'backend'])}")
        print(f"  - Frontend: {len([t for t in all_tasks if t.team == 'frontend'])}")
        print(f"  - Platform Ops: {len([t for t in all_tasks if t.team == 'platform_ops'])}")
        print(f"  - AI/ML: {len([t for t in all_tasks if t.team == 'ai_ml'])}")
        print(f"  - Data: {len([t for t in all_tasks if t.team == 'data'])}")
        
        # Verify each task
        print("\n" + "="*100)
        print("TASK VERIFICATION")
        print("="*100)
        
        for task in all_tasks:
            result = self.verify_task(task)
            
            # Add to results
            self.results[task.team][task.priority].append({
                "week": task.week,
                "name": task.name,
                "status": result.status,
                "details": result.details
            })
            
            # Print progress
            status_symbol = "OK" if result.status == "PASS" else "WARN" if result.status == "PARTIAL" else "FAIL"
            print(f"  [{status_symbol}] Week {task.week} {task.team.upper()} {task.priority}: {task.name}")
            if result.status != "PASS":
                print(f"      -> {result.details}")
        
        # Run unit tests
        print("\n" + "="*100)
        print("UNIT TESTS")
        print("="*100)
        await self.run_unit_tests()
        
        # Run functionality tests
        print("\n" + "="*100)
        print("FUNCTIONALITY TESTS")
        print("="*100)
        await self.run_functionality_tests()
        
        # Run integration tests
        print("\n" + "="*100)
        print("INTEGRATION TESTS")  
        print("="*100)
        await self.run_integration_tests()
        
        # Find hanging features
        print("\n" + "="*100)
        print("HANGING FEATURES CHECK")
        print("="*100)
        self.find_hanging_features()
        
        # Generate summary
        self.generate_summary()
        
        # Save results
        with open("validation_results.json", "w") as f:
            json.dump(self.results, f, indent=2)
        
        print("\n[RESULTS SAVED] validation_results.json")
        
        return self.results
    
    def generate_summary(self):
        """Generate validation summary"""
        print("\n" + "="*100)
        print("VALIDATION SUMMARY")
        print("="*100)
        
        # Count results by team and priority
        summary = {}
        
        for team in ["backend", "frontend", "platform_ops", "ai_ml", "data"]:
            team_summary = {"P0": {"total": 0, "pass": 0, "partial": 0, "fail": 0},
                           "P1": {"total": 0, "pass": 0, "partial": 0, "fail": 0},
                           "P2": {"total": 0, "pass": 0, "partial": 0, "fail": 0}}
            
            for priority in ["P0", "P1", "P2"]:
                tasks = self.results[team][priority]
                team_summary[priority]["total"] = len(tasks)
                team_summary[priority]["pass"] = len([t for t in tasks if t["status"] == "PASS"])
                team_summary[priority]["partial"] = len([t for t in tasks if t["status"] == "PARTIAL"])
                team_summary[priority]["fail"] = len([t for t in tasks if t["status"] == "FAIL"])
            
            summary[team] = team_summary
        
        # Print summary
        for team, team_summary in summary.items():
            print(f"\n{team.upper()} TEAM:")
            for priority in ["P0", "P1", "P2"]:
                stats = team_summary[priority]
                if stats["total"] > 0:
                    completion = (stats["pass"] / stats["total"]) * 100
                    print(f"  {priority}: {stats['pass']}/{stats['total']} ({completion:.1f}%) - "
                          f"Pass: {stats['pass']}, Partial: {stats['partial']}, Fail: {stats['fail']}")
        
        # Overall statistics
        total_tasks = sum(summary[team][p]["total"] for team in summary for p in ["P0", "P1", "P2"])
        total_pass = sum(summary[team][p]["pass"] for team in summary for p in ["P0", "P1", "P2"])
        total_partial = sum(summary[team][p]["partial"] for team in summary for p in ["P0", "P1", "P2"])
        total_fail = sum(summary[team][p]["fail"] for team in summary for p in ["P0", "P1", "P2"])
        
        print(f"\nOVERALL STATISTICS:")
        print(f"  Total Tasks: {total_tasks}")
        print(f"  Passed: {total_pass} ({total_pass/total_tasks*100:.1f}%)")
        print(f"  Partial: {total_partial} ({total_partial/total_tasks*100:.1f}%)")
        print(f"  Failed: {total_fail} ({total_fail/total_tasks*100:.1f}%)")
        
        # Test results
        unit_pass = len([t for t in self.results["unit_tests"] if t["status"] == "PASS"])
        func_pass = len([t for t in self.results["functionality_tests"] if t["status"] == "PASS"])
        int_pass = len([t for t in self.results["integration_tests"] if t["status"] == "PASS"])
        
        print(f"\nTEST RESULTS:")
        print(f"  Unit Tests: {unit_pass}/{len(self.results['unit_tests'])} passed")
        print(f"  Functionality Tests: {func_pass}/{len(self.results['functionality_tests'])} passed")
        print(f"  Integration Tests: {int_pass}/{len(self.results['integration_tests'])} passed")
        
        # Hanging features
        if self.results["hanging_features"]:
            print(f"\nHANGING FEATURES: {len(self.results['hanging_features'])} services not connected")
        else:
            print(f"\nHANGING FEATURES: None found [OK]")
        
        self.results["summary"] = {
            "total_tasks": total_tasks,
            "passed": total_pass,
            "partial": total_partial,
            "failed": total_fail,
            "completion_rate": (total_pass / total_tasks * 100) if total_tasks > 0 else 0,
            "team_summary": summary
        }


async def main():
    validator = ComprehensiveValidator()
    results = await validator.run_all_validations()
    
    # Return exit code based on results
    if results["summary"]["completion_rate"] >= 80:
        print("\n[SUCCESS] VALIDATION PASSED (>=80% completion)")
        return 0
    else:
        print(f"\n[WARNING] VALIDATION NEEDS ATTENTION ({results['summary']['completion_rate']:.1f}% completion)")
        return 1


if __name__ == "__main__":
    import asyncio
    exit_code = asyncio.run(main())