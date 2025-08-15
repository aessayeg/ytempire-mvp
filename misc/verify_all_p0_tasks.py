"""
Comprehensive verification of ALL P0 (CRITICAL) tasks across all teams
Based on Week 2 Execution Plan requirements
"""

import os
import sys
import json
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Any, Tuple
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class P0TaskVerifier:
    """Verify all P0 critical tasks implementation"""
    
    def __init__(self):
        self.project_root = Path("C:/Users/Hp/projects/ytempire-mvp")
        self.verification_results = {
            "timestamp": datetime.now().isoformat(),
            "teams": {},
            "summary": {
                "total_tasks": 0,
                "implemented": 0,
                "tested": 0,
                "integrated": 0,
                "fully_complete": 0
            }
        }
        
        # Define all P0 tasks with verification criteria
        self.p0_tasks = {
            "BACKEND_TEAM": [
                {
                    "name": "Scaling Video Pipeline to 100/day",
                    "files_to_check": [
                        "backend/app/core/celery_app.py",
                        "backend/app/workers",
                        "backend/app/services/video_generation.py"
                    ],
                    "key_features": ["Celery", "Worker auto-scaling", "Connection pooling"]
                },
                {
                    "name": "API Performance Optimization",
                    "files_to_check": [
                        "backend/app/core/cache.py",
                        "backend/app/api/v1/endpoints/api_optimization.py",
                        "backend/app/core/performance_enhanced.py"
                    ],
                    "key_features": ["Redis caching", "N+1 query fixes", "<300ms response"]
                },
                {
                    "name": "Multi-Channel Architecture",
                    "files_to_check": [
                        "backend/app/models/channel.py",
                        "backend/app/services/channel_manager.py",
                        "backend/app/api/v1/endpoints/channels.py"
                    ],
                    "key_features": ["Channel isolation", "Quota management", "5+ channels"]
                },
                {
                    "name": "Subscription & Billing APIs",
                    "files_to_check": [
                        "backend/app/api/v1/endpoints/payment.py",
                        "backend/app/services/subscription_service.py",
                        "backend/app/models/subscription.py"
                    ],
                    "key_features": ["Tier management", "Usage billing", "Invoice generation"]
                },
                {
                    "name": "Batch Operations Implementation",
                    "files_to_check": [
                        "backend/app/api/v1/endpoints/batch.py",
                        "backend/app/services/batch_processor.py"
                    ],
                    "key_features": ["Batch generation", "Status tracking", "50+ items"]
                },
                {
                    "name": "Real-time Collaboration APIs",
                    "files_to_check": [
                        "backend/app/services/websocket_manager.py",
                        "backend/app/api/v1/endpoints/collaboration.py",
                        "backend/app/services/notification_service.py"
                    ],
                    "key_features": ["WebSocket rooms", "Real-time notifications", "Progress tracking"]
                },
                {
                    "name": "Advanced Video Processing Features",
                    "files_to_check": [
                        "backend/app/services/video_processor.py",
                        "backend/app/api/v1/endpoints/video_processing.py",
                        "backend/app/services/advanced_video_processing.py"
                    ],
                    "key_features": ["Quality enhancement", "Multi-format", "3x throughput"]
                },
                {
                    "name": "Advanced Analytics Pipeline",
                    "files_to_check": [
                        "backend/app/api/v1/endpoints/advanced_analytics.py",
                        "backend/app/services/analytics_service.py",
                        "backend/app/services/real_time_analytics.py"
                    ],
                    "key_features": ["Real-time streaming", "Predictive models", "GDPR"]
                },
                {
                    "name": "Advanced N8N Workflows",
                    "files_to_check": [
                        "infrastructure/n8n/workflows",
                        "backend/app/services/workflow_manager.py"
                    ],
                    "key_features": ["Multi-channel posting", "Intelligent scheduling", "10+ workflows"]
                },
                {
                    "name": "Multi-Account YouTube Management",
                    "files_to_check": [
                        "backend/app/services/youtube_multi_account.py",
                        "backend/app/api/v1/endpoints/youtube_accounts.py",
                        "backend/app/services/youtube_manager.py"
                    ],
                    "key_features": ["15 account rotation", "Quota tracking", "Automatic failover"]
                }
            ],
            "FRONTEND_TEAM": [
                {
                    "name": "Beta User UI Refinements",
                    "files_to_check": [
                        "frontend/src/components/Navigation",
                        "frontend/src/components/Help",
                        "frontend/src/styles"
                    ],
                    "key_features": ["Navigation redesign", "Tooltips", "Mobile responsive"]
                },
                {
                    "name": "Dashboard Enhancement & Analytics",
                    "files_to_check": [
                        "frontend/src/components/Dashboard",
                        "frontend/src/components/Charts",
                        "frontend/src/components/Widgets"
                    ],
                    "key_features": ["Advanced charts", "Customizable widgets", "Real-time updates"]
                },
                {
                    "name": "Channel Dashboard Implementation",
                    "files_to_check": [
                        "frontend/src/pages/ChannelDashboard",
                        "frontend/src/components/ChannelOverview"
                    ],
                    "key_features": ["Channel overview", "Channel metrics", "Video history"]
                },
                {
                    "name": "Advanced Channel Management",
                    "files_to_check": [
                        "frontend/src/components/ChannelManager",
                        "frontend/src/components/Channels/ChannelBulkOperations.tsx"
                    ],
                    "key_features": ["Bulk operations", "Templates", "Health dashboard"]
                },
                {
                    "name": "Real-time Monitoring Dashboard",
                    "files_to_check": [
                        "frontend/src/components/RealTimeMonitor",
                        "frontend/src/components/CostTracker"
                    ],
                    "key_features": ["Live generation monitor", "Cost tracking", "System health"]
                },
                {
                    "name": "Analytics Dashboard Implementation",
                    "files_to_check": [
                        "frontend/src/pages/Analytics",
                        "frontend/src/components/Analytics"
                    ],
                    "key_features": ["Revenue tracking", "Performance metrics", "Channel comparison"]
                },
                {
                    "name": "Beta User Journey Optimization",
                    "files_to_check": [
                        "frontend/src/analytics",
                        "frontend/src/components/UserJourney"
                    ],
                    "key_features": ["Session recording", "Pain points", "User testing"]
                },
                {
                    "name": "Beta User Onboarding Flow",
                    "files_to_check": [
                        "frontend/src/components/Onboarding",
                        "frontend/src/pages/Welcome"
                    ],
                    "key_features": ["5-step journey", "Welcome screens", "Interactive tutorials"]
                }
            ],
            "PLATFORM_OPS_TEAM": [
                {
                    "name": "Production Deployment",
                    "files_to_check": [
                        "infrastructure/deployment",
                        "infrastructure/scripts/deploy.sh",
                        ".github/workflows"
                    ],
                    "key_features": ["Deployment checklist", "Monitoring", "Automated backups"]
                },
                {
                    "name": "Scaling Infrastructure",
                    "files_to_check": [
                        "infrastructure/kubernetes",
                        "infrastructure/scaling"
                    ],
                    "key_features": ["50+ concurrent users", "Resource optimization", "Load balancing"]
                },
                {
                    "name": "High Availability Implementation",
                    "files_to_check": [
                        "infrastructure/ha",
                        "infrastructure/database/replication.yml"
                    ],
                    "key_features": ["Database replication", "Service redundancy", "Automatic failover"]
                },
                {
                    "name": "Blue-Green Deployment",
                    "files_to_check": [
                        "infrastructure/deployment/blue_green.yml",
                        "infrastructure/scripts/switch_environment.sh"
                    ],
                    "key_features": ["Parallel environments", "Traffic switching", "Zero-downtime"]
                },
                {
                    "name": "CI/CD Pipeline Maturation",
                    "files_to_check": [
                        ".github/workflows",
                        "infrastructure/ci",
                        "infrastructure/security/scanning"
                    ],
                    "key_features": ["Security scanning", "Performance gates", "Canary deployments"]
                },
                {
                    "name": "Observability Platform",
                    "files_to_check": [
                        "infrastructure/monitoring",
                        "infrastructure/logging",
                        "infrastructure/tracing"
                    ],
                    "key_features": ["Distributed tracing", "Centralized logging", "SLO tracking"]
                },
                {
                    "name": "Security Hardening Sprint",
                    "files_to_check": [
                        "infrastructure/security",
                        "infrastructure/waf",
                        "backend/app/core/security.py"
                    ],
                    "key_features": ["Security audit", "WAF", "DDoS protection"]
                },
                {
                    "name": "Production Security Hardening",
                    "files_to_check": [
                        "backend/app/middleware/rate_limiter.py",
                        "backend/app/core/audit_logger.py"
                    ],
                    "key_features": ["Rate limiting", "Audit logging", "Security scan"]
                },
                {
                    "name": "Identity & Access Management",
                    "files_to_check": [
                        "backend/app/core/auth.py",
                        "backend/app/services/rbac.py",
                        "backend/app/services/sso.py"
                    ],
                    "key_features": ["MFA", "RBAC", "SSO", "Zero-trust"]
                },
                {
                    "name": "Data Encryption Implementation",
                    "files_to_check": [
                        "backend/app/core/encryption.py",
                        "infrastructure/security/tls"
                    ],
                    "key_features": ["Database encryption", "Field-level encryption", "TLS"]
                },
                {
                    "name": "Comprehensive Test Automation",
                    "files_to_check": [
                        "tests/e2e",
                        "tests/visual",
                        "tests/integration"
                    ],
                    "key_features": ["50+ E2E tests", "Visual regression", "80% coverage"]
                },
                {
                    "name": "Beta User Acceptance Testing",
                    "files_to_check": [
                        "tests/uat",
                        "tests/scenarios"
                    ],
                    "key_features": ["Test scenarios", "E2E suite", "Issue documentation"]
                },
                {
                    "name": "Quality Metrics Framework",
                    "files_to_check": [
                        "infrastructure/metrics",
                        "infrastructure/dashboards/quality"
                    ],
                    "key_features": ["KPIs definition", "Defect tracking", "Quality dashboards"]
                }
            ],
            "AI_ML_TEAM": [
                {
                    "name": "Multi-Model Orchestration",
                    "files_to_check": [
                        "ml-pipeline/src/orchestrator.py",
                        "ml-pipeline/src/model_selector.py",
                        "backend/app/services/ml_orchestration.py"
                    ],
                    "key_features": ["Performance tracking", "Dynamic selection", "Cost routing"]
                },
                {
                    "name": "Model Optimization Sprint",
                    "files_to_check": [
                        "ml-pipeline/src/model_optimizer.py",
                        "ml-pipeline/src/quantization.py"
                    ],
                    "key_features": ["Model quantization", "Inference batching", "50% cost reduction"]
                },
                {
                    "name": "Advanced Script Generation",
                    "files_to_check": [
                        "backend/app/services/script_generation.py",
                        "ml-pipeline/src/script_generator.py",
                        "backend/app/api/v1/endpoints/script_generation.py"
                    ],
                    "key_features": ["Style transfer", "Tone modulation", "20 variations"]
                },
                {
                    "name": "Advanced Trend Prediction",
                    "files_to_check": [
                        "ml-pipeline/src/trend_predictor.py",
                        "backend/app/services/trend_analysis.py"
                    ],
                    "key_features": ["TikTok/Reddit integration", "Ensemble modeling", "85% accuracy"]
                }
            ],
            "DATA_TEAM": [
                {
                    "name": "Beta User Analytics Platform",
                    "files_to_check": [
                        "backend/app/services/user_analytics.py",
                        "backend/app/api/v1/endpoints/analytics.py"
                    ],
                    "key_features": ["Behavior tracking", "Performance dashboards", "Daily reports"]
                },
                {
                    "name": "Real-time Analytics Pipeline",
                    "files_to_check": [
                        "backend/app/services/real_time_analytics.py",
                        "infrastructure/streaming"
                    ],
                    "key_features": ["Streaming analytics", "Real-time aggregations", "Live dashboards"]
                },
                {
                    "name": "Real-time Feature Pipeline",
                    "files_to_check": [
                        "ml-pipeline/src/feature_pipeline.py",
                        "backend/app/services/feature_store.py"
                    ],
                    "key_features": ["Streaming ingestion", "Real-time computation", "Feature monitoring"]
                },
                {
                    "name": "Real-time Processing Scale-up",
                    "files_to_check": [
                        "backend/app/services/data_pipeline.py",
                        "infrastructure/data_processing"
                    ],
                    "key_features": ["Parallel processing", "10x volume handling"]
                },
                {
                    "name": "Beta User Success Metrics",
                    "files_to_check": [
                        "backend/app/services/success_metrics.py",
                        "backend/app/api/v1/endpoints/metrics.py"
                    ],
                    "key_features": ["KPIs definition", "Success dashboards", "Daily reports"]
                },
                {
                    "name": "Business Intelligence Dashboard",
                    "files_to_check": [
                        "backend/app/api/v1/endpoints/business_intelligence.py",
                        "backend/app/services/bi_service.py"
                    ],
                    "key_features": ["Executive dashboard", "Financial reports", "User analytics"]
                }
            ]
        }
    
    def check_file_exists(self, file_path: str) -> bool:
        """Check if a file or directory exists"""
        full_path = self.project_root / file_path
        return full_path.exists()
    
    def check_files_for_task(self, files: List[str]) -> Tuple[int, List[str]]:
        """Check which files exist for a task"""
        found_files = []
        for file_path in files:
            if self.check_file_exists(file_path):
                found_files.append(file_path)
            else:
                # Check with common extensions
                for ext in ['.py', '.ts', '.tsx', '.js', '.jsx', '.yml', '.yaml']:
                    if self.check_file_exists(f"{file_path}{ext}"):
                        found_files.append(f"{file_path}{ext}")
                        break
        
        return len(found_files), found_files
    
    def check_integration(self, task_name: str, team: str) -> bool:
        """Check if task is integrated into main application"""
        # Check if APIs are registered in main router
        if "API" in task_name or "endpoint" in task_name.lower():
            api_router_path = self.project_root / "backend/app/api/v1/api.py"
            if api_router_path.exists():
                with open(api_router_path, 'r') as f:
                    content = f.read()
                    # Check for relevant imports/registrations
                    keywords = task_name.lower().replace(" ", "_").split("_")[:2]
                    for keyword in keywords:
                        if keyword in content.lower():
                            return True
        
        # Check if frontend components are used in main app
        if team == "FRONTEND_TEAM":
            app_path = self.project_root / "frontend/src/App.tsx"
            if app_path.exists():
                with open(app_path, 'r') as f:
                    content = f.read()
                    task_keywords = task_name.lower().replace(" ", "").replace("-", "")
                    if any(kw in content.lower() for kw in task_keywords.split()[:2]):
                        return True
        
        return False
    
    def check_tests(self, task_name: str) -> bool:
        """Check if tests exist for the task"""
        test_dirs = [
            "tests",
            "backend/tests",
            "frontend/src/__tests__",
            "ml-pipeline/tests"
        ]
        
        task_keywords = task_name.lower().replace(" ", "_").split("_")[:2]
        
        for test_dir in test_dirs:
            test_path = self.project_root / test_dir
            if test_path.exists():
                for root, dirs, files in os.walk(test_path):
                    for file in files:
                        if any(kw in file.lower() for kw in task_keywords):
                            return True
        
        return False
    
    def verify_task(self, task: Dict, team: str) -> Dict[str, Any]:
        """Verify a single task"""
        task_name = task["name"]
        files_to_check = task["files_to_check"]
        key_features = task["key_features"]
        
        # Check implementation
        found_count, found_files = self.check_files_for_task(files_to_check)
        implementation_score = (found_count / len(files_to_check)) * 100 if files_to_check else 0
        
        # Check integration
        is_integrated = self.check_integration(task_name, team)
        
        # Check tests
        has_tests = self.check_tests(task_name)
        
        # Determine status
        is_implemented = implementation_score >= 50
        is_fully_complete = is_implemented and is_integrated and has_tests
        
        return {
            "name": task_name,
            "implementation_score": implementation_score,
            "files_found": f"{found_count}/{len(files_to_check)}",
            "found_files": found_files[:3],  # Show first 3 files
            "is_implemented": is_implemented,
            "is_integrated": is_integrated,
            "has_tests": has_tests,
            "is_fully_complete": is_fully_complete,
            "key_features": key_features
        }
    
    def verify_all_teams(self):
        """Verify all P0 tasks across all teams"""
        logger.info("="*80)
        logger.info("P0 (CRITICAL) TASKS VERIFICATION - ALL TEAMS")
        logger.info("="*80)
        
        for team, tasks in self.p0_tasks.items():
            logger.info(f"\n{'='*60}")
            logger.info(f"TEAM: {team}")
            logger.info(f"{'='*60}")
            
            team_results = {
                "tasks": [],
                "total": len(tasks),
                "implemented": 0,
                "integrated": 0,
                "tested": 0,
                "fully_complete": 0
            }
            
            for task in tasks:
                result = self.verify_task(task, team)
                team_results["tasks"].append(result)
                
                if result["is_implemented"]:
                    team_results["implemented"] += 1
                if result["is_integrated"]:
                    team_results["integrated"] += 1
                if result["has_tests"]:
                    team_results["tested"] += 1
                if result["is_fully_complete"]:
                    team_results["fully_complete"] += 1
                
                # Log task status
                status_icon = "✅" if result["is_fully_complete"] else "⚠️" if result["is_implemented"] else "❌"
                logger.info(f"\n{status_icon} {result['name']}")
                logger.info(f"   Files: {result['files_found']} found")
                logger.info(f"   Implementation: {result['implementation_score']:.0f}%")
                logger.info(f"   Integrated: {'Yes' if result['is_integrated'] else 'No'}")
                logger.info(f"   Tested: {'Yes' if result['has_tests'] else 'No'}")
            
            self.verification_results["teams"][team] = team_results
            
            # Update global summary
            self.verification_results["summary"]["total_tasks"] += team_results["total"]
            self.verification_results["summary"]["implemented"] += team_results["implemented"]
            self.verification_results["summary"]["integrated"] += team_results["integrated"]
            self.verification_results["summary"]["tested"] += team_results["tested"]
            self.verification_results["summary"]["fully_complete"] += team_results["fully_complete"]
            
            # Log team summary
            logger.info(f"\n{team} Summary:")
            logger.info(f"  Total Tasks: {team_results['total']}")
            logger.info(f"  Implemented: {team_results['implemented']}/{team_results['total']} ({team_results['implemented']/team_results['total']*100:.1f}%)")
            logger.info(f"  Integrated: {team_results['integrated']}/{team_results['total']}")
            logger.info(f"  Tested: {team_results['tested']}/{team_results['total']}")
            logger.info(f"  Fully Complete: {team_results['fully_complete']}/{team_results['total']} ({team_results['fully_complete']/team_results['total']*100:.1f}%)")
    
    def generate_report(self):
        """Generate final verification report"""
        summary = self.verification_results["summary"]
        
        logger.info("\n" + "="*80)
        logger.info("OVERALL P0 TASKS VERIFICATION SUMMARY")
        logger.info("="*80)
        
        # Overall metrics
        total = summary["total_tasks"]
        implemented_pct = (summary["implemented"] / total * 100) if total > 0 else 0
        integrated_pct = (summary["integrated"] / total * 100) if total > 0 else 0
        tested_pct = (summary["tested"] / total * 100) if total > 0 else 0
        complete_pct = (summary["fully_complete"] / total * 100) if total > 0 else 0
        
        logger.info(f"\nTotal P0 Tasks: {total}")
        logger.info(f"Implemented: {summary['implemented']}/{total} ({implemented_pct:.1f}%)")
        logger.info(f"Integrated: {summary['integrated']}/{total} ({integrated_pct:.1f}%)")
        logger.info(f"Tested: {summary['tested']}/{total} ({tested_pct:.1f}%)")
        logger.info(f"Fully Complete: {summary['fully_complete']}/{total} ({complete_pct:.1f}%)")
        
        # Team breakdown
        logger.info("\n" + "-"*60)
        logger.info("TEAM BREAKDOWN:")
        for team, results in self.verification_results["teams"].items():
            completion_rate = (results["fully_complete"] / results["total"] * 100) if results["total"] > 0 else 0
            logger.info(f"\n{team}:")
            logger.info(f"  Tasks: {results['total']}")
            logger.info(f"  Complete: {results['fully_complete']} ({completion_rate:.1f}%)")
            
            # Show incomplete tasks
            incomplete_tasks = [t["name"] for t in results["tasks"] if not t["is_fully_complete"]]
            if incomplete_tasks:
                logger.info(f"  Incomplete: {', '.join(incomplete_tasks[:3])}" + 
                          (" ..." if len(incomplete_tasks) > 3 else ""))
        
        # Critical issues
        logger.info("\n" + "-"*60)
        logger.info("CRITICAL ISSUES:")
        
        critical_missing = []
        for team, results in self.verification_results["teams"].items():
            for task in results["tasks"]:
                if not task["is_implemented"]:
                    critical_missing.append(f"{team}: {task['name']}")
        
        if critical_missing:
            logger.warning(f"\n⚠️ {len(critical_missing)} P0 tasks not implemented:")
            for item in critical_missing[:10]:
                logger.warning(f"  - {item}")
            if len(critical_missing) > 10:
                logger.warning(f"  ... and {len(critical_missing) - 10} more")
        else:
            logger.info("✅ All P0 tasks have at least partial implementation")
        
        # Save detailed report
        report_file = self.project_root / "misc" / "p0_tasks_verification_report.json"
        with open(report_file, 'w') as f:
            json.dump(self.verification_results, f, indent=2, default=str)
        logger.info(f"\n📄 Detailed report saved to: {report_file}")
        
        # Final verdict
        if complete_pct >= 80:
            logger.info("\n" + "="*80)
            logger.info("✅ P0 TASKS SUFFICIENTLY COMPLETE FOR BETA!")
            logger.info(f"   {complete_pct:.1f}% of critical tasks fully implemented")
            logger.info("="*80)
        elif complete_pct >= 60:
            logger.warning("\n" + "="*80)
            logger.warning("⚠️ P0 TASKS PARTIALLY COMPLETE")
            logger.warning(f"   Only {complete_pct:.1f}% fully complete")
            logger.warning("   Additional work needed for production")
            logger.warning("="*80)
        else:
            logger.error("\n" + "="*80)
            logger.error("❌ P0 TASKS INCOMPLETE")
            logger.error(f"   Only {complete_pct:.1f}% fully complete")
            logger.error("   Significant work required")
            logger.error("="*80)
        
        return complete_pct


def main():
    """Main execution"""
    verifier = P0TaskVerifier()
    verifier.verify_all_teams()
    completion_rate = verifier.generate_report()
    
    return 0 if completion_rate >= 60 else 1


if __name__ == "__main__":
    sys.exit(main())