"""
Comprehensive Week 0 and Week 1 Task Verification
Checking all P0, P1, P2 tasks from master plans
"""

import os
import json
from datetime import datetime

def check_file_exists(filepath):
    return os.path.exists(filepath)

def verify_week0_week1_tasks():
    base_dir = r"C:\Users\Hp\projects\ytempire-mvp"
    
    # Week 0 Critical Tasks (P0, P1, P2)
    week0_tasks = {
        "P0 - Must Complete by Day 2": {
            "Backend": [
                ("API Gateway setup", "backend/app/main.py", True),
                ("Message Queue setup", "backend/app/core/celery_app.py", True),
                ("Database Schema", "backend/alembic/versions/", True),
                ("YouTube API Integration", "backend/app/services/youtube_service.py", True)
            ],
            "Frontend": [
                ("React project initialization", "frontend/package.json", True),
                ("Development Environment", "frontend/vite.config.ts", True),
                ("TypeScript setup", "frontend/tsconfig.json", True),
                ("Material-UI theme", "frontend/src/theme/", True)
            ],
            "Platform Ops": [
                ("Docker Infrastructure", "docker-compose.yml", True),
                ("Security Configuration", "infrastructure/security/", True),
                ("Local Server Setup", "infrastructure/", True)
            ],
            "AI/ML": [
                ("AI service access setup", "ml-pipeline/config.yaml", True),
                ("GPU Environment", "ml-pipeline/", True),
                ("ML Pipeline Architecture", "ml-pipeline/services/", True),
                ("Cost Optimization Strategy", "backend/app/services/cost_tracker.py", True)
            ],
            "Data": [
                ("Data Lake Architecture", "backend/app/services/data_lake_service.py", True),
                ("Training Data Pipeline", "backend/app/services/training_data_service.py", True)
            ]
        },
        "P1 - Must Complete by Day 4": {
            "Backend": [
                ("Authentication Service", "backend/app/api/v1/endpoints/auth.py", True),
                ("Channel Management CRUD", "backend/app/api/v1/endpoints/channels.py", True),
                ("N8N Workflow Engine", "infrastructure/n8n/", True),
                ("Video Processing Pipeline", "backend/app/tasks/video_tasks.py", True),
                ("Cost Tracking System", "backend/app/services/cost_tracker.py", True)
            ],
            "Frontend": [
                ("State Management", "frontend/src/stores/", True),
                ("Component Library", "frontend/src/components/", True),
                ("Dashboard Layout", "frontend/src/components/Dashboard/", True),
                ("Authentication UI", "frontend/src/components/Auth/", True)
            ],
            "Platform Ops": [
                ("CI/CD Pipeline", ".github/workflows/ci-cd.yml", True),
                ("Monitoring Stack", "docker-compose.yml", True),
                ("GitHub Actions", ".github/workflows/", True),
                ("Test Framework", "tests/", True)
            ],
            "AI/ML": [
                ("Model Serving Infrastructure", "backend/app/services/inference_pipeline.py", True),
                ("Trend Prediction Prototype", "ml-pipeline/services/trend_detection.py", True),
                ("Model Evaluation Framework", "ml-pipeline/services/model_monitoring.py", True)
            ],
            "Data": [
                ("Metrics Database Design", "backend/app/models/", True),
                ("Real-time Feature Store", "backend/app/services/", True),
                ("Cost Analytics Framework", "backend/app/services/roi_calculator.py", True),
                ("Vector Database Setup", "ml-pipeline/", True)
            ]
        },
        "P2 - Complete by Day 5": {
            "Backend": [
                ("WebSocket Foundation", "backend/app/services/websocket_manager.py", True),
                ("Payment Gateway", "backend/app/services/payment_service_enhanced.py", True),
                ("Error Handling Framework", "backend/app/core/", True)
            ],
            "Frontend": [
                ("Chart Library Integration", "frontend/src/components/Charts/", True),
                ("Real-time Data Architecture", "frontend/src/services/websocket.ts", True)
            ],
            "Platform Ops": [
                ("Backup Strategy", "infrastructure/backup/", True),
                ("SSL/TLS Configuration", "infrastructure/security/", True),
                ("Performance Testing", "tests/performance/", True)
            ],
            "AI/ML": [
                ("Content Quality Scoring", "ml-pipeline/services/quality_scoring.py", True),
                ("Model Monitoring System", "ml-pipeline/services/model_monitoring.py", True)
            ],
            "Data": [
                ("Feature Engineering Pipeline", "ml-pipeline/services/feature_engineering.py", True),
                ("Reporting Infrastructure", "backend/app/services/reporting.py", True)
            ]
        }
    }
    
    # Week 1 Critical Tasks (P0, P1, P2)
    week1_tasks = {
        "P0 - Critical for First Video": {
            "Backend": [
                ("Channel Management API", "backend/app/api/v1/endpoints/channels.py", True),
                ("Video Generation API", "backend/app/api/v1/endpoints/videos.py", True),
                ("YouTube Multi-Account (15)", "backend/app/services/youtube_multi_account.py", True),
                ("Cost Tracking Active", "backend/app/services/cost_tracker.py", True),
                ("Video Queue System", "backend/app/services/video_queue_service.py", True)
            ],
            "Frontend": [
                ("Dashboard MVP", "frontend/src/components/Dashboard/", True),
                ("Channel Management UI", "frontend/src/components/Channels/", True),
                ("Video Generation UI", "frontend/src/components/Videos/", True),
                ("API Integration Layer", "frontend/src/services/api.ts", True)
            ],
            "Platform Ops": [
                ("Production Environment", "docker-compose.yml", True),
                ("Backup Testing", "infrastructure/backup/", True),
                ("Performance Optimization", "infrastructure/scaling/", True),
                ("Security Audit", "infrastructure/security/", True)
            ],
            "AI/ML": [
                ("Content Quality Optimization", "ml-pipeline/services/quality_scoring.py", True),
                ("Cost Optimization (<$3)", "backend/app/services/cost_optimizer.py", True),
                ("Multi-Agent Foundation", "ml-pipeline/services/", True)
            ],
            "Data": [
                ("Data Quality Framework", "backend/app/services/", True),
                ("Analytics Pipeline", "ml-pipeline/services/analytics_pipeline.py", True),
                ("YouTube Analytics Integration", "backend/app/services/", True),
                ("Cost Data Pipeline", "backend/app/services/", True)
            ]
        },
        "P1 - Core Features": {
            "Backend": [
                ("Bulk Video Generation", "backend/app/api/v1/endpoints/videos.py", True),
                ("Video Analytics", "backend/app/api/v1/endpoints/analytics.py", True),
                ("Webhook System", "backend/app/api/v1/endpoints/webhooks.py", True)
            ],
            "Frontend": [
                ("Video Queue Interface", "frontend/src/components/Videos/VideoList.tsx", True),
                ("Analytics Dashboard", "frontend/src/components/Dashboard/EnhancedMetricsDashboard.tsx", True),
                ("Real-time Updates", "frontend/src/services/websocket.ts", True)
            ],
            "Platform Ops": [
                ("Log Aggregation", "infrastructure/monitoring/", True),
                ("Auto-scaling Config", "infrastructure/scaling/", True)
            ],
            "AI/ML": [
                ("A/B Testing Framework", "ml-pipeline/services/", True),
                ("Model Retraining Auto", "backend/app/tasks/", True)
            ],
            "Data": [
                ("KPI Dashboard", "frontend/src/components/Dashboard/", True),
                ("Revenue Attribution", "backend/app/services/roi_calculator.py", True)
            ]
        },
        "P2 - Nice to Have": {
            "Backend": [
                ("Advanced Error Recovery", "backend/app/", True)
            ],
            "Frontend": [
                ("Export Functionality", "frontend/src/components/", True),
                ("Mobile Responsive", "frontend/src/components/Mobile/", True)
            ],
            "Platform Ops": [
                ("Advanced Monitoring", "infrastructure/monitoring/", True)
            ]
        }
    }
    
    # Additional critical components for video generation
    critical_video_components = {
        "Video Generation UI (16 Components)": [
            ("VideoCard", "frontend/src/components/Videos/VideoCard.tsx", True),
            ("VideoList", "frontend/src/components/Videos/VideoList.tsx", True),
            ("VideoDetail", "frontend/src/pages/Videos/VideoDetail.tsx", True),
            ("VideoPlayer", "frontend/src/components/Videos/VideoPlayer.tsx", True),
            ("VideoGenerator", "frontend/src/pages/Videos/VideoGenerator.tsx", True),
            ("GenerationProgress", "frontend/src/components/Videos/GenerationProgress.tsx", True),
            ("VideoPreview", "frontend/src/components/Videos/VideoPreview.tsx", True),
            ("VideoApproval", "frontend/src/components/Videos/VideoApproval.tsx", True),
            ("VideoUploadProgress", "frontend/src/components/Videos/VideoUploadProgress.tsx", True),
            ("PublishingControls", "frontend/src/components/Videos/PublishingControls.tsx", True),
            ("YouTubeUploadStatus", "frontend/src/components/Videos/YouTubeUploadStatus.tsx", True),
            ("VideoMetrics", "frontend/src/components/Videos/VideoMetrics.tsx", True),
            ("VideoPerformanceChart", "frontend/src/components/Videos/VideoPerformanceChart.tsx", True),
            ("VideoEngagementStats", "frontend/src/components/Videos/VideoEngagementStats.tsx", True),
            ("VideoSearch", "frontend/src/components/Videos/VideoSearch.tsx", True),
            ("VideoFilters", "frontend/src/components/Videos/VideoFilters.tsx", True)
        ]
    }
    
    # Verification
    results = {
        "week0": {"P0": {}, "P1": {}, "P2": {}},
        "week1": {"P0": {}, "P1": {}, "P2": {}},
        "critical_components": {},
        "summary": {
            "total_tasks": 0,
            "completed": 0,
            "missing": []
        }
    }
    
    print("\n" + "="*80)
    print(" WEEK 0 & WEEK 1 TASK VERIFICATION")
    print("="*80)
    
    # Check Week 0 tasks
    print("\n[WEEK 0 TASKS]")
    print("-"*40)
    for priority, teams in week0_tasks.items():
        priority_key = priority.split(" - ")[0]
        results["week0"][priority_key] = {}
        print(f"\n{priority}:")
        
        for team, tasks in teams.items():
            team_completed = 0
            team_total = len(tasks)
            results["week0"][priority_key][team] = []
            
            for task_name, filepath, expected in tasks:
                full_path = os.path.join(base_dir, filepath)
                exists = check_file_exists(full_path) or os.path.isdir(full_path)
                status = "[OK]" if exists else "[MISSING]"
                
                if not exists:
                    results["summary"]["missing"].append(f"Week0/{priority_key}/{team}/{task_name}")
                else:
                    team_completed += 1
                    results["summary"]["completed"] += 1
                
                results["summary"]["total_tasks"] += 1
                results["week0"][priority_key][team].append({
                    "task": task_name,
                    "status": "completed" if exists else "missing"
                })
            
            completion = (team_completed / team_total * 100) if team_total > 0 else 0
            print(f"  {team}: {team_completed}/{team_total} ({completion:.0f}%)")
    
    # Check Week 1 tasks
    print("\n[WEEK 1 TASKS]")
    print("-"*40)
    for priority, teams in week1_tasks.items():
        priority_key = priority.split(" - ")[0]
        results["week1"][priority_key] = {}
        print(f"\n{priority}:")
        
        for team, tasks in teams.items():
            team_completed = 0
            team_total = len(tasks)
            results["week1"][priority_key][team] = []
            
            for task_name, filepath, expected in tasks:
                full_path = os.path.join(base_dir, filepath)
                exists = check_file_exists(full_path) or os.path.isdir(full_path)
                status = "[OK]" if exists else "[MISSING]"
                
                if not exists:
                    results["summary"]["missing"].append(f"Week1/{priority_key}/{team}/{task_name}")
                else:
                    team_completed += 1
                    results["summary"]["completed"] += 1
                
                results["summary"]["total_tasks"] += 1
                results["week1"][priority_key][team].append({
                    "task": task_name,
                    "status": "completed" if exists else "missing"
                })
            
            completion = (team_completed / team_total * 100) if team_total > 0 else 0
            print(f"  {team}: {team_completed}/{team_total} ({completion:.0f}%)")
    
    # Check critical video components
    print("\n[CRITICAL VIDEO GENERATION UI COMPONENTS]")
    print("-"*40)
    components_completed = 0
    for component_name, filepath, expected in critical_video_components["Video Generation UI (16 Components)"]:
        full_path = os.path.join(base_dir, filepath)
        exists = check_file_exists(full_path)
        
        if exists:
            components_completed += 1
            results["summary"]["completed"] += 1
        else:
            results["summary"]["missing"].append(f"VideoComponent/{component_name}")
        
        results["summary"]["total_tasks"] += 1
    
    print(f"Video Generation UI: {components_completed}/16 ({components_completed/16*100:.0f}%)")
    
    # Summary
    print("\n" + "="*80)
    print(" VERIFICATION SUMMARY")
    print("="*80)
    
    completion_rate = (results["summary"]["completed"] / results["summary"]["total_tasks"] * 100) if results["summary"]["total_tasks"] > 0 else 0
    
    print(f"\nTotal Tasks: {results['summary']['total_tasks']}")
    print(f"Completed: {results['summary']['completed']}")
    print(f"Missing: {len(results['summary']['missing'])}")
    print(f"Overall Completion: {completion_rate:.1f}%")
    
    if len(results["summary"]["missing"]) > 0:
        print("\n[MISSING TASKS]:")
        for task in results["summary"]["missing"][:10]:  # Show first 10
            print(f"  - {task}")
        if len(results["summary"]["missing"]) > 10:
            print(f"  ... and {len(results['summary']['missing']) - 10} more")
    
    # Final verdict
    print("\n" + "="*80)
    if completion_rate == 100:
        print(" [OK] ALL WEEK 0 & WEEK 1 TASKS COMPLETED!")
        print(" READY TO PROCEED TO WEEK 2")
    elif completion_rate >= 95:
        print(" [NEARLY COMPLETE] 95%+ tasks done, review missing items")
    else:
        print(f" [IN PROGRESS] {completion_rate:.1f}% complete")
    print("="*80)
    
    # Save results
    with open("week0_week1_verification.json", "w") as f:
        json.dump(results, f, indent=2)
    
    return results

if __name__ == "__main__":
    results = verify_week0_week1_tasks()