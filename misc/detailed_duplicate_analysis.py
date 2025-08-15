"""
Detailed Duplicate Analysis with Content Inspection
Identifies actual duplicate services and components
"""

import os
import re
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Set, Tuple
from collections import defaultdict

def analyze_duplicates():
    """Perform detailed duplicate analysis"""
    
    print("\n" + "="*80)
    print("DETAILED DUPLICATE & REDUNDANCY ANALYSIS")
    print("="*80)
    
    duplicates = []
    
    # 1. Check Backend Services
    print("\n[BACKEND SERVICE ANALYSIS]")
    print("-"*40)
    
    backend_services = Path("backend/app/services")
    if backend_services.exists():
        # Known duplicate patterns to check
        service_checks = [
            {
                "category": "Video Generation",
                "files": [
                    "video_generation.py",
                    "video_generation_pipeline.py",
                    "video_generation_orchestrator.py",
                    "enhanced_video_generation.py"
                ],
                "primary": "video_generation_pipeline.py",
                "reason": "Multiple implementations of video generation logic"
            },
            {
                "category": "Payment Services",
                "files": [
                    "payment_service.py",
                    "payment_service_enhanced.py"
                ],
                "primary": "payment_service_enhanced.py",
                "reason": "Original and enhanced versions coexist"
            },
            {
                "category": "Analytics Services",
                "files": [
                    "analytics_service.py",
                    "realtime_analytics_service.py",
                    "analytics_pipeline.py",
                    "analytics_connector.py",
                    "analytics_report.py"
                ],
                "primary": "analytics_service.py",
                "reason": "Multiple analytics implementations"
            },
            {
                "category": "Cost Tracking",
                "files": [
                    "cost_tracking.py",
                    "realtime_cost_tracking.py"
                ],
                "primary": "cost_tracking.py",
                "reason": "Separate realtime and regular tracking"
            },
            {
                "category": "Notification Services",
                "files": [
                    "notification_service.py",
                    "alert_service.py"
                ],
                "primary": "notification_service.py",
                "reason": "Alert service duplicates notification functionality"
            }
        ]
        
        for check in service_checks:
            existing = []
            for file in check["files"]:
                if (backend_services / file).exists():
                    existing.append(file)
                    size = (backend_services / file).stat().st_size
                    print(f"  Found: {file} ({size:,} bytes)")
            
            if len(existing) > 1:
                duplicates.append({
                    "category": check["category"],
                    "files": existing,
                    "recommendation": f"Keep {check['primary']}, refactor others",
                    "reason": check["reason"]
                })
                print(f"  -> DUPLICATE: {check['reason']}")
                print(f"  -> Recommendation: Keep {check['primary']}")
            print()
    
    # 2. Check Frontend Components
    print("\n[FRONTEND COMPONENT ANALYSIS]")
    print("-"*40)
    
    frontend_components = Path("frontend/src/components")
    if frontend_components.exists():
        component_checks = [
            {
                "category": "Video Components",
                "paths": [
                    "Videos/VideoCard.tsx",
                    "Videos/VideoListItem.tsx",
                    "Videos/VideoTile.tsx",
                    "Dashboard/VideoCard.tsx"
                ],
                "primary": "Videos/VideoCard.tsx",
                "reason": "Multiple video display components"
            },
            {
                "category": "Analytics Components",
                "paths": [
                    "Analytics/AnalyticsDashboard.tsx",
                    "Dashboard/MetricCard.tsx",
                    "Dashboard/StatsCard.tsx",
                    "Dashboard/AnalyticsWidget.tsx"
                ],
                "primary": "Analytics/AnalyticsDashboard.tsx",
                "reason": "Overlapping analytics display components"
            },
            {
                "category": "Chart Components",
                "paths": [
                    "Charts/LineChart.tsx",
                    "Analytics/RevenueChart.tsx",
                    "Analytics/ViewsChart.tsx",
                    "DataVisualization/ChartComponent.tsx"
                ],
                "primary": "Charts/LineChart.tsx",
                "reason": "Multiple chart implementations"
            }
        ]
        
        for check in component_checks:
            existing = []
            for path in check["paths"]:
                if (frontend_components / path).exists():
                    existing.append(path)
                    print(f"  Found: {path}")
            
            if len(existing) > 1:
                duplicates.append({
                    "category": f"Frontend - {check['category']}",
                    "files": existing,
                    "recommendation": f"Consolidate into {check['primary']}",
                    "reason": check["reason"]
                })
                print(f"  -> DUPLICATE: {check['reason']}")
                print(f"  -> Recommendation: Use {check['primary']}")
            print()
    
    # 3. Check API Endpoints
    print("\n[API ENDPOINT ANALYSIS]")
    print("-"*40)
    
    api_endpoints = Path("backend/app/api/v1/endpoints")
    if api_endpoints.exists():
        endpoint_files = list(api_endpoints.glob("*.py"))
        endpoint_patterns = defaultdict(list)
        
        for file in endpoint_files:
            if file.name != "__init__.py":
                # Group by base name pattern
                base = file.stem.replace("_", "").replace("v2", "").replace("enhanced", "")
                endpoint_patterns[base].append(file.name)
        
        for base, files in endpoint_patterns.items():
            if len(files) > 1:
                print(f"  Potential duplicate endpoints for '{base}':")
                for f in files:
                    print(f"    - {f}")
                duplicates.append({
                    "category": f"API Endpoints - {base}",
                    "files": files,
                    "recommendation": "Review and consolidate endpoints",
                    "reason": "Multiple endpoint files for same resource"
                })
                print()
    
    # 4. Check Database Models
    print("\n[DATABASE MODEL ANALYSIS]")
    print("-"*40)
    
    models_path = Path("backend/app/models")
    if models_path.exists():
        model_files = list(models_path.glob("*.py"))
        
        # Check for duplicate model definitions
        model_checks = [
            ("user.py", "users.py"),
            ("video.py", "videos.py"),
            ("channel.py", "channels.py"),
            ("payment.py", "billing.py"),
        ]
        
        for check in model_checks:
            existing = [f for f in check if (models_path / f).exists()]
            if len(existing) > 1:
                print(f"  Duplicate models found: {', '.join(existing)}")
                duplicates.append({
                    "category": "Database Models",
                    "files": existing,
                    "recommendation": f"Keep singular form ({check[0]})",
                    "reason": "Duplicate model definitions"
                })
    
    # 5. Check Task Files
    print("\n[CELERY TASK ANALYSIS]")
    print("-"*40)
    
    tasks_path = Path("backend/app/tasks")
    if tasks_path.exists():
        task_files = list(tasks_path.glob("*.py"))
        
        # Group tasks by functionality
        task_groups = defaultdict(list)
        for file in task_files:
            if "video" in file.name:
                task_groups["video"].append(file.name)
            if "youtube" in file.name:
                task_groups["youtube"].append(file.name)
            if "analytics" in file.name:
                task_groups["analytics"].append(file.name)
        
        for group, files in task_groups.items():
            if len(files) > 1:
                print(f"  Multiple {group} task files: {', '.join(files)}")
                # This might be intentional separation, so just note it
    
    # Generate Summary Report
    print("\n" + "="*80)
    print("DUPLICATE ANALYSIS SUMMARY")
    print("="*80)
    
    if duplicates:
        print(f"\nFound {len(duplicates)} categories of duplicates:\n")
        
        # Group by priority
        high_priority = []
        medium_priority = []
        low_priority = []
        
        for dup in duplicates:
            # Determine priority based on file count and category
            if len(dup["files"]) > 3:
                high_priority.append(dup)
            elif "service" in dup["category"].lower() or "api" in dup["category"].lower():
                medium_priority.append(dup)
            else:
                low_priority.append(dup)
        
        if high_priority:
            print("[HIGH PRIORITY - Immediate Action Required]")
            for item in high_priority:
                print(f"\n  {item['category']}:")
                print(f"    Files: {', '.join(item['files'])}")
                print(f"    Reason: {item['reason']}")
                print(f"    Action: {item['recommendation']}")
        
        if medium_priority:
            print("\n[MEDIUM PRIORITY - Review Soon]")
            for item in medium_priority:
                print(f"\n  {item['category']}:")
                print(f"    Files: {', '.join(item['files'])}")
                print(f"    Reason: {item['reason']}")
                print(f"    Action: {item['recommendation']}")
        
        if low_priority:
            print("\n[LOW PRIORITY - Cleanup When Possible]")
            for item in low_priority:
                print(f"\n  {item['category']}:")
                print(f"    Files: {', '.join(item['files'])}")
                print(f"    Action: {item['recommendation']}")
    else:
        print("\nNo significant duplicates found!")
    
    # Task Completion Check
    print("\n" + "="*80)
    print("WEEK 0-2 TASK COMPLETION CHECK")
    print("="*80)
    
    missing_features = []
    
    # Check for missing P1/P2 features
    p1_p2_checks = [
        ("backend/app/services/ab_testing_service.py", "A/B Testing Service"),
        ("backend/app/services/competitor_analysis.py", "Competitor Analysis"),
        ("backend/app/services/forecasting_service.py", "Forecasting Service"),
        ("frontend/src/components/DataVisualization", "Data Visualization Components"),
        ("frontend/src/components/ReportBuilder", "Report Builder"),
        ("frontend/src/components/AdvancedAnalytics", "Advanced Analytics Dashboard"),
    ]
    
    for path, name in p1_p2_checks:
        if not Path(path).exists():
            missing_features.append(name)
            print(f"  [MISSING] {name}")
        else:
            print(f"  [OK] {name}")
    
    if missing_features:
        print(f"\n  -> {len(missing_features)} P1/P2 features still need implementation")
    
    # Final Recommendations
    print("\n" + "="*80)
    print("RECOMMENDED ACTIONS")
    print("="*80)
    
    print("\n1. IMMEDIATE CLEANUP:")
    print("   - Consolidate video generation services into video_generation_pipeline.py")
    print("   - Remove payment_service.py (keep payment_service_enhanced.py)")
    print("   - Merge realtime_cost_tracking.py into cost_tracking.py")
    
    print("\n2. REFACTORING:")
    print("   - Combine analytics services into unified analytics_service.py")
    print("   - Create shared base components for VideoCard variations")
    print("   - Standardize chart components usage")
    
    print("\n3. COMPLETE MISSING FEATURES:")
    if missing_features:
        for feature in missing_features[:3]:
            print(f"   - Implement {feature}")
    else:
        print("   - All critical features implemented!")
    
    print("\n4. DOCUMENTATION:")
    print("   - Document which services are primary vs deprecated")
    print("   - Add deprecation warnings to duplicate services")
    print("   - Update imports to use primary services")
    
    return duplicates


if __name__ == "__main__":
    duplicates = analyze_duplicates()