#!/usr/bin/env python3
"""
Real Project Metrics Collector
Gathers actual statistics from the YTEmpire MVP codebase
"""

import os
import json
import subprocess
from datetime import datetime
from pathlib import Path
import re
from typing import Dict, List, Any

class RealMetricsCollector:
    def __init__(self):
        self.project_root = Path.cwd()
        self.metrics = {
            "timestamp": datetime.now().isoformat(),
            "project_name": "YTEmpire MVP",
            "actual_metrics": {},
            "codebase_stats": {},
            "git_stats": {},
            "test_coverage": {},
            "services_status": {},
            "implementation_status": {}
        }
        
    def count_lines_of_code(self) -> Dict:
        """Count actual lines of code in the project"""
        print("📊 Analyzing codebase...")
        
        stats = {
            "python": {"files": 0, "lines": 0, "blank": 0, "comment": 0},
            "typescript": {"files": 0, "lines": 0, "blank": 0, "comment": 0},
            "javascript": {"files": 0, "lines": 0, "blank": 0, "comment": 0},
            "total": {"files": 0, "lines": 0}
        }
        
        extensions = {
            ".py": "python",
            ".ts": "typescript",
            ".tsx": "typescript",
            ".js": "javascript",
            ".jsx": "javascript"
        }
        
        for root, dirs, files in os.walk(self.project_root):
            # Skip node_modules, venv, __pycache__ etc
            dirs[:] = [d for d in dirs if d not in [
                'node_modules', 'venv', '__pycache__', '.git', 
                'dist', 'build', '.pytest_cache', '.next'
            ]]
            
            for file in files:
                ext = Path(file).suffix
                if ext in extensions:
                    file_path = Path(root) / file
                    lang = extensions[ext]
                    stats[lang]["files"] += 1
                    stats["total"]["files"] += 1
                    
                    try:
                        with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                            lines = f.readlines()
                            stats[lang]["lines"] += len(lines)
                            stats["total"]["lines"] += len(lines)
                            
                            for line in lines:
                                stripped = line.strip()
                                if not stripped:
                                    stats[lang]["blank"] += 1
                                elif stripped.startswith(('#', '//', '/*', '*')):
                                    stats[lang]["comment"] += 1
                    except:
                        pass
                        
        return stats
        
    def get_git_statistics(self) -> Dict:
        """Get actual git statistics"""
        print("📈 Gathering git statistics...")
        
        stats = {
            "total_commits": 0,
            "contributors": 0,
            "branches": 0,
            "last_commit": None,
            "files_changed": 0
        }
        
        try:
            # Get total commits
            result = subprocess.run(
                ["git", "rev-list", "--count", "HEAD"],
                capture_output=True, text=True, cwd=self.project_root
            )
            if result.returncode == 0:
                stats["total_commits"] = int(result.stdout.strip())
            
            # Get contributors
            result = subprocess.run(
                ["git", "shortlog", "-sn"],
                capture_output=True, text=True, cwd=self.project_root
            )
            if result.returncode == 0:
                stats["contributors"] = len(result.stdout.strip().split('\n'))
            
            # Get branches
            result = subprocess.run(
                ["git", "branch", "-r"],
                capture_output=True, text=True, cwd=self.project_root
            )
            if result.returncode == 0:
                stats["branches"] = len(result.stdout.strip().split('\n'))
            
            # Get last commit date
            result = subprocess.run(
                ["git", "log", "-1", "--format=%ci"],
                capture_output=True, text=True, cwd=self.project_root
            )
            if result.returncode == 0:
                stats["last_commit"] = result.stdout.strip()
                
            # Get files changed in last commit
            result = subprocess.run(
                ["git", "diff", "--stat", "HEAD~1"],
                capture_output=True, text=True, cwd=self.project_root
            )
            if result.returncode == 0:
                lines = result.stdout.strip().split('\n')
                if lines:
                    stats["files_changed"] = len(lines) - 1  # Exclude summary line
                    
        except Exception as e:
            print(f"  ⚠️  Error getting git stats: {e}")
            
        return stats
        
    def check_services_status(self) -> Dict:
        """Check which services are actually running"""
        print("🔍 Checking services status...")
        
        services = {
            "backend": {"port": 8000, "running": False},
            "frontend": {"port": 3000, "running": False},
            "postgresql": {"port": 5432, "running": False},
            "redis": {"port": 6379, "running": False},
            "grafana": {"port": 3001, "running": False},
            "prometheus": {"port": 9090, "running": False}
        }
        
        import socket
        for service, config in services.items():
            sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            sock.settimeout(1)
            result = sock.connect_ex(('localhost', config['port']))
            sock.close()
            config['running'] = result == 0
            
        return services
        
    def analyze_test_coverage(self) -> Dict:
        """Get actual test coverage if available"""
        print("🧪 Analyzing test coverage...")
        
        coverage = {
            "backend": {"has_tests": False, "test_files": 0, "test_count": 0},
            "frontend": {"has_tests": False, "test_files": 0, "test_count": 0},
            "total_test_files": 0
        }
        
        # Count backend test files
        backend_tests = list(Path("backend/tests").glob("**/test_*.py")) if Path("backend/tests").exists() else []
        backend_tests.extend(list(Path("tests").glob("**/test_*.py")) if Path("tests").exists() else [])
        coverage["backend"]["test_files"] = len(backend_tests)
        coverage["backend"]["has_tests"] = len(backend_tests) > 0
        
        # Count frontend test files
        frontend_tests = list(Path("frontend/src").glob("**/*.test.{ts,tsx,js,jsx}")) if Path("frontend/src").exists() else []
        coverage["frontend"]["test_files"] = len(frontend_tests)
        coverage["frontend"]["has_tests"] = len(frontend_tests) > 0
        
        coverage["total_test_files"] = coverage["backend"]["test_files"] + coverage["frontend"]["test_files"]
        
        # Try to get actual test count from pytest
        try:
            result = subprocess.run(
                ["pytest", "--collect-only", "-q"],
                capture_output=True, text=True, cwd=self.project_root / "backend"
            )
            if result.returncode == 0:
                # Parse pytest output for test count
                lines = result.stdout.strip().split('\n')
                for line in lines:
                    if 'test' in line.lower():
                        coverage["backend"]["test_count"] += 1
        except:
            pass
            
        return coverage
        
    def check_implementation_status(self) -> Dict:
        """Check what has actually been implemented"""
        print("✅ Checking implementation status...")
        
        status = {
            "docker": {
                "docker_compose_exists": Path("docker-compose.yml").exists() or Path("docker-compose.yaml").exists(),
                "dockerfile_exists": Path("Dockerfile").exists() or Path("backend/Dockerfile").exists()
            },
            "backend": {
                "main_app_exists": Path("backend/app/main.py").exists(),
                "api_endpoints": 0,
                "models_defined": 0,
                "services_implemented": 0
            },
            "frontend": {
                "package_json_exists": Path("frontend/package.json").exists(),
                "components_count": 0,
                "pages_count": 0
            },
            "ml_pipeline": {
                "models_implemented": 0,
                "config_exists": Path("ml-pipeline/config.yaml").exists()
            },
            "database": {
                "migrations_exist": Path("backend/alembic").exists() or Path("backend/migrations").exists(),
                "models_defined": 0
            }
        }
        
        # Count API endpoints
        api_path = Path("backend/app/api/v1/endpoints")
        if api_path.exists():
            status["backend"]["api_endpoints"] = len(list(api_path.glob("*.py")))
            
        # Count models
        models_path = Path("backend/app/models")
        if models_path.exists():
            status["backend"]["models_defined"] = len(list(models_path.glob("*.py")))
            
        # Count services
        services_path = Path("backend/app/services")
        if services_path.exists():
            status["backend"]["services_implemented"] = len(list(services_path.glob("*.py")))
            
        # Count frontend components
        components_path = Path("frontend/src/components")
        if components_path.exists():
            status["frontend"]["components_count"] = len(list(components_path.glob("**/*.tsx")))
            
        # Count frontend pages
        pages_path = Path("frontend/src/pages")
        if pages_path.exists():
            status["frontend"]["pages_count"] = len(list(pages_path.glob("**/*.tsx")))
            
        # Count ML models
        ml_path = Path("ml-pipeline/src")
        if ml_path.exists():
            status["ml_pipeline"]["models_implemented"] = len(list(ml_path.glob("*.py")))
            
        return status
        
    def generate_report(self):
        """Generate comprehensive real metrics report"""
        print("\n" + "="*60)
        print("REAL PROJECT METRICS REPORT".center(60))
        print("="*60 + "\n")
        
        # Collect all metrics
        self.metrics["codebase_stats"] = self.count_lines_of_code()
        self.metrics["git_stats"] = self.get_git_statistics()
        self.metrics["services_status"] = self.check_services_status()
        self.metrics["test_coverage"] = self.analyze_test_coverage()
        self.metrics["implementation_status"] = self.check_implementation_status()
        
        # Calculate actual vs planned
        actual_vs_planned = {
            "videos_generated": {
                "planned": 12,
                "actual": 0,  # No actual videos generated yet
                "status": "Not Started"
            },
            "cost_per_video": {
                "planned": "$2.10",
                "actual": "N/A",
                "status": "No videos generated"
            },
            "test_coverage": {
                "planned": "87.3%",
                "actual": f"{self.metrics['test_coverage']['total_test_files']} test files",
                "status": "Tests exist but coverage not measured"
            },
            "api_endpoints": {
                "planned": "15+",
                "actual": self.metrics["implementation_status"]["backend"]["api_endpoints"],
                "status": "Partially Implemented"
            },
            "youtube_accounts": {
                "planned": 15,
                "actual": 0,
                "status": "Not Integrated"
            },
            "beta_users": {
                "planned": 5,
                "actual": 0,
                "status": "No users yet"
            }
        }
        
        self.metrics["actual_vs_planned"] = actual_vs_planned
        
        # Print summary
        print("📊 CODEBASE STATISTICS:")
        print(f"  Total Files: {self.metrics['codebase_stats']['total']['files']}")
        print(f"  Total Lines: {self.metrics['codebase_stats']['total']['lines']:,}")
        print(f"  Python: {self.metrics['codebase_stats']['python']['files']} files, {self.metrics['codebase_stats']['python']['lines']:,} lines")
        print(f"  TypeScript/React: {self.metrics['codebase_stats']['typescript']['files']} files, {self.metrics['codebase_stats']['typescript']['lines']:,} lines")
        
        print("\n📈 GIT STATISTICS:")
        print(f"  Total Commits: {self.metrics['git_stats']['total_commits']}")
        print(f"  Contributors: {self.metrics['git_stats']['contributors']}")
        print(f"  Last Commit: {self.metrics['git_stats']['last_commit']}")
        
        print("\n🔍 SERVICES STATUS:")
        running_services = sum(1 for s in self.metrics['services_status'].values() if s['running'])
        print(f"  Running Services: {running_services}/{len(self.metrics['services_status'])}")
        for service, status in self.metrics['services_status'].items():
            icon = "✅" if status['running'] else "❌"
            print(f"    {icon} {service}: {'Running' if status['running'] else 'Not Running'}")
        
        print("\n✅ IMPLEMENTATION STATUS:")
        print(f"  Backend API Endpoints: {self.metrics['implementation_status']['backend']['api_endpoints']}")
        print(f"  Backend Services: {self.metrics['implementation_status']['backend']['services_implemented']}")
        print(f"  Frontend Components: {self.metrics['implementation_status']['frontend']['components_count']}")
        print(f"  Frontend Pages: {self.metrics['implementation_status']['frontend']['pages_count']}")
        print(f"  Docker Setup: {'Yes' if self.metrics['implementation_status']['docker']['docker_compose_exists'] else 'No'}")
        
        print("\n🧪 TEST COVERAGE:")
        print(f"  Backend Test Files: {self.metrics['test_coverage']['backend']['test_files']}")
        print(f"  Frontend Test Files: {self.metrics['test_coverage']['frontend']['test_files']}")
        print(f"  Total Test Files: {self.metrics['test_coverage']['total_test_files']}")
        
        print("\n⚠️  ACTUAL vs PLANNED:")
        for metric, data in actual_vs_planned.items():
            print(f"  {metric.replace('_', ' ').title()}:")
            print(f"    Planned: {data['planned']}")
            print(f"    Actual: {data['actual']}")
            print(f"    Status: {data['status']}")
        
        # Save report
        report_file = f"real_metrics_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        with open(report_file, 'w') as f:
            json.dump(self.metrics, f, indent=2)
            
        print(f"\n💾 Full report saved to: {report_file}")
        
        print("\n" + "="*60)
        print("REALITY CHECK COMPLETE".center(60))
        print("="*60)
        
        print("\n🔍 SUMMARY:")
        print("  The metrics in the documentation are PLANNED TARGETS, not actual achievements.")
        print("  The project has good structure and initial implementation but is not yet operational.")
        print("  No videos have been generated, no users onboarded, and services are not running.")
        print("\n  This is a work-in-progress MVP with solid planning documentation.")

def main():
    collector = RealMetricsCollector()
    collector.generate_report()

if __name__ == "__main__":
    main()