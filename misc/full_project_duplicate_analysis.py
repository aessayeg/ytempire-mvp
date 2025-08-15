"""
Full Project Verification and Duplicate Analysis
Checks Week 0-2 completion and identifies duplicate/redundant components
"""

import os
import json
import hashlib
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Set, Tuple
from collections import defaultdict
import re

class ProjectAnalyzer:
    def __init__(self):
        self.project_root = Path(".")
        self.duplicates = defaultdict(list)
        self.similar_files = defaultdict(list)
        self.task_status = {
            "week0": {"P0": [], "P1": [], "P2": []},
            "week1": {"P0": [], "P1": [], "P2": []},
            "week2": {"P0": [], "P1": [], "P2": []}
        }
        self.duplicate_patterns = []
        
    def analyze_project(self):
        """Main analysis function"""
        print("\n" + "="*80)
        print("YTEMPIRE MVP - FULL PROJECT VERIFICATION & DUPLICATE ANALYSIS")
        print("="*80)
        print(f"Analysis Date: {datetime.now().isoformat()}")
        
        results = {
            "timestamp": datetime.now().isoformat(),
            "duplicates": {},
            "similar_files": {},
            "task_completion": {},
            "recommendations": []
        }
        
        # 1. Find duplicate services
        print("\n[1/5] Analyzing Backend Services...")
        self.analyze_backend_services()
        
        # 2. Find duplicate components
        print("[2/5] Analyzing Frontend Components...")
        self.analyze_frontend_components()
        
        # 3. Check task completion
        print("[3/5] Verifying Week 0-2 Task Completion...")
        self.verify_task_completion()
        
        # 4. Find similar/redundant files
        print("[4/5] Identifying Similar Files...")
        self.find_similar_files()
        
        # 5. Generate recommendations
        print("[5/5] Generating Recommendations...")
        recommendations = self.generate_recommendations()
        
        # Compile results
        results["duplicates"] = dict(self.duplicates)
        results["similar_files"] = dict(self.similar_files)
        results["task_completion"] = self.task_status
        results["recommendations"] = recommendations
        
        return results
    
    def analyze_backend_services(self):
        """Analyze backend services for duplicates"""
        services_path = self.project_root / "backend" / "app" / "services"
        
        if not services_path.exists():
            return
        
        service_groups = defaultdict(list)
        
        # Group services by functionality
        patterns = {
            "video": ["video", "generation", "pipeline", "orchestrat"],
            "payment": ["payment", "billing", "subscription", "stripe", "invoice"],
            "analytics": ["analytics", "metrics", "reporting", "stats"],
            "youtube": ["youtube", "channel", "upload"],
            "ai": ["ai", "openai", "gpt", "claude", "script", "thumbnail"],
            "notification": ["notification", "alert", "email", "sms"],
            "auth": ["auth", "security", "jwt", "oauth"],
            "cost": ["cost", "tracking", "budget", "expense"],
            "batch": ["batch", "bulk", "queue"],
            "websocket": ["websocket", "ws", "realtime", "socket"],
            "cache": ["cache", "redis", "memory"],
        }
        
        for file in services_path.glob("*.py"):
            if file.name == "__init__.py":
                continue
                
            file_lower = file.stem.lower()
            
            # Categorize by pattern
            for category, keywords in patterns.items():
                if any(keyword in file_lower for keyword in keywords):
                    service_groups[category].append(file.name)
        
        # Check for duplicates within categories
        for category, files in service_groups.items():
            if len(files) > 1:
                # Check if files are actually duplicates
                potential_duplicates = self.check_service_duplicates(services_path, files)
                if potential_duplicates:
                    self.duplicates[f"backend_services_{category}"] = potential_duplicates
    
    def check_service_duplicates(self, base_path: Path, files: List[str]) -> List[Dict]:
        """Check if services are actual duplicates by analyzing content"""
        duplicates = []
        
        # Special known duplicate patterns
        duplicate_pairs = [
            ("video_generation.py", "video_generation_pipeline.py", "video_generation_orchestrator.py"),
            ("payment_service.py", "payment_service_enhanced.py"),
            ("cost_tracking.py", "realtime_cost_tracking.py"),
            ("analytics_service.py", "realtime_analytics_service.py"),
        ]
        
        for pair in duplicate_pairs:
            existing = [f for f in pair if f in files]
            if len(existing) > 1:
                # Analyze if they're truly duplicates
                analysis = self.analyze_file_similarity(base_path, existing)
                if analysis["similarity"] > 0.6:  # 60% similarity threshold
                    duplicates.append({
                        "files": existing,
                        "similarity": analysis["similarity"],
                        "recommendation": analysis["recommendation"]
                    })
        
        return duplicates
    
    def analyze_file_similarity(self, base_path: Path, files: List[str]) -> Dict:
        """Analyze similarity between files"""
        contents = []
        functions = []
        
        for file in files:
            file_path = base_path / file
            if file_path.exists():
                try:
                    with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                        content = f.read()
                        contents.append(content)
                        
                        # Extract function names
                        func_pattern = r'def\s+(\w+)\s*\('
                        class_pattern = r'class\s+(\w+)'
                        
                        funcs = re.findall(func_pattern, content)
                        classes = re.findall(class_pattern, content)
                        functions.append(set(funcs + classes))
                except:
                    pass
        
        if len(functions) < 2:
            return {"similarity": 0, "recommendation": "Unable to analyze"}
        
        # Calculate similarity based on shared functions/classes
        common = functions[0].intersection(*functions[1:])
        total = set().union(*functions)
        
        similarity = len(common) / len(total) if total else 0
        
        # Generate recommendation
        if similarity > 0.8:
            recommendation = "High duplication - merge into single service"
        elif similarity > 0.6:
            recommendation = "Significant overlap - consider refactoring shared code"
        elif similarity > 0.4:
            recommendation = "Some overlap - review for consolidation opportunities"
        else:
            recommendation = "Different purposes - keep separate"
        
        return {
            "similarity": similarity,
            "common_functions": list(common),
            "recommendation": recommendation
        }
    
    def analyze_frontend_components(self):
        """Analyze frontend components for duplicates"""
        components_path = self.project_root / "frontend" / "src" / "components"
        
        if not components_path.exists():
            return
        
        component_groups = defaultdict(list)
        
        # Group components by type
        patterns = {
            "video": ["Video", "Player", "Generation", "Upload"],
            "dashboard": ["Dashboard", "Stats", "Analytics", "Metrics"],
            "channel": ["Channel", "YouTube"],
            "auth": ["Auth", "Login", "Register", "SignUp"],
            "form": ["Form", "Input", "Field"],
            "modal": ["Modal", "Dialog", "Popup"],
            "chart": ["Chart", "Graph", "Plot"],
            "table": ["Table", "Grid", "List"],
            "card": ["Card", "Tile", "Panel"],
        }
        
        for file in components_path.rglob("*.tsx"):
            if "test" in file.name or "spec" in file.name:
                continue
            
            for category, keywords in patterns.items():
                if any(keyword in file.stem for keyword in keywords):
                    component_groups[category].append(file.relative_to(components_path))
        
        # Check for duplicates
        for category, files in component_groups.items():
            if len(files) > 1:
                similar = self.check_component_duplicates(components_path, files)
                if similar:
                    self.duplicates[f"frontend_components_{category}"] = similar
    
    def check_component_duplicates(self, base_path: Path, files: List[Path]) -> List[Dict]:
        """Check for duplicate React components"""
        duplicates = []
        
        # Known duplicate patterns
        duplicate_patterns = [
            ("VideoCard", "VideoListItem", "VideoTile"),
            ("DashboardStats", "StatsCard", "MetricsCard"),
            ("ChannelCard", "ChannelListItem", "ChannelTile"),
            ("LoginForm", "AuthForm", "SignInForm"),
        ]
        
        file_names = [f.stem for f in files]
        
        for pattern in duplicate_patterns:
            matches = [f for f in files if any(p in f.stem for p in pattern)]
            if len(matches) > 1:
                duplicates.append({
                    "components": [str(f) for f in matches],
                    "type": "Similar naming/functionality",
                    "recommendation": "Review for consolidation"
                })
        
        return duplicates
    
    def verify_task_completion(self):
        """Verify Week 0-2 task completion based on planning documents"""
        
        # Week 0 P0 Tasks (Critical Foundation)
        week0_p0 = {
            "Backend API": self.check_exists("backend/app/main.py"),
            "Database Models": self.check_exists("backend/app/models"),
            "Authentication": self.check_exists("backend/app/core/security.py"),
            "Frontend Setup": self.check_exists("frontend/src/App.tsx"),
            "Docker Config": self.check_exists("docker-compose.yml"),
        }
        
        # Week 1 P0 Tasks (Core Features)
        week1_p0 = {
            "Video Generation": self.check_multiple([
                "backend/app/services/video_generation_pipeline.py",
                "backend/app/services/script_generation.py"
            ]),
            "YouTube Integration": self.check_exists("backend/app/services/youtube_service.py"),
            "Dashboard": self.check_exists("frontend/src/pages/Dashboard"),
            "Channel Management": self.check_exists("backend/app/services/channel_manager.py"),
        }
        
        # Week 2 P0 Tasks (Scaling & Advanced)
        week2_p0 = {
            "Multi-Channel": self.check_exists("backend/app/services/youtube_multi_account.py"),
            "Batch Processing": self.check_exists("backend/app/services/batch_processing.py"),
            "Subscription System": self.check_exists("backend/app/services/subscription_service.py"),
            "WebSocket": self.check_exists("backend/app/services/websocket_manager.py"),
            "Cost Tracking": self.check_exists("backend/app/services/cost_tracking.py"),
        }
        
        # Week 1 P1 Tasks
        week1_p1 = {
            "Analytics": self.check_exists("backend/app/services/analytics_service.py"),
            "Notifications": self.check_exists("backend/app/services/notification_service.py"),
            "Video Editor": self.check_exists("frontend/src/components/VideoEditor"),
        }
        
        # Week 2 P1 Tasks
        week2_p1 = {
            "Advanced Analytics": self.check_exists("frontend/src/components/AdvancedAnalytics"),
            "A/B Testing": self.check_exists("backend/app/services/ab_testing_service.py"),
            "Competitor Analysis": self.check_exists("backend/app/services/competitor_analysis.py"),
        }
        
        # Week 2 P2 Tasks
        week2_p2 = {
            "Data Visualization": self.check_exists("frontend/src/components/DataVisualization"),
            "Report Builder": self.check_exists("frontend/src/components/ReportBuilder"),
            "Forecasting": self.check_exists("backend/app/services/forecasting_service.py"),
        }
        
        self.task_status = {
            "week0": {
                "P0": self.calculate_completion(week0_p0),
                "P1": {"status": "N/A", "tasks": []},
                "P2": {"status": "N/A", "tasks": []}
            },
            "week1": {
                "P0": self.calculate_completion(week1_p0),
                "P1": self.calculate_completion(week1_p1),
                "P2": {"status": "N/A", "tasks": []}
            },
            "week2": {
                "P0": self.calculate_completion(week2_p0),
                "P1": self.calculate_completion(week2_p1),
                "P2": self.calculate_completion(week2_p2)
            }
        }
    
    def check_exists(self, path: str) -> bool:
        """Check if a file or directory exists"""
        return (self.project_root / path).exists()
    
    def check_multiple(self, paths: List[str]) -> bool:
        """Check if all paths exist"""
        return all(self.check_exists(p) for p in paths)
    
    def calculate_completion(self, tasks: Dict[str, bool]) -> Dict:
        """Calculate task completion percentage"""
        completed = sum(1 for v in tasks.values() if v)
        total = len(tasks)
        
        return {
            "completed": completed,
            "total": total,
            "percentage": (completed / total * 100) if total > 0 else 0,
            "tasks": [{"name": k, "completed": v} for k, v in tasks.items()]
        }
    
    def find_similar_files(self):
        """Find files with similar names that might be duplicates"""
        
        # Backend services
        backend_services = list((self.project_root / "backend" / "app" / "services").glob("*.py"))
        
        # Group by similarity
        for i, file1 in enumerate(backend_services):
            for file2 in backend_services[i+1:]:
                similarity = self.calculate_name_similarity(file1.stem, file2.stem)
                if similarity > 0.7:  # 70% name similarity
                    self.similar_files["backend_services"].append({
                        "file1": file1.name,
                        "file2": file2.name,
                        "similarity": similarity
                    })
        
        # Frontend components
        frontend_components = list((self.project_root / "frontend" / "src" / "components").rglob("*.tsx"))
        
        for i, file1 in enumerate(frontend_components):
            for file2 in frontend_components[i+1:]:
                similarity = self.calculate_name_similarity(file1.stem, file2.stem)
                if similarity > 0.7:
                    self.similar_files["frontend_components"].append({
                        "file1": str(file1.relative_to(self.project_root / "frontend" / "src" / "components")),
                        "file2": str(file2.relative_to(self.project_root / "frontend" / "src" / "components")),
                        "similarity": similarity
                    })
    
    def calculate_name_similarity(self, name1: str, name2: str) -> float:
        """Calculate similarity between two names"""
        # Normalize names
        n1 = name1.lower().replace("_", "").replace("-", "")
        n2 = name2.lower().replace("_", "").replace("-", "")
        
        # Check if one contains the other
        if n1 in n2 or n2 in n1:
            return 0.8
        
        # Calculate Levenshtein-like similarity
        common = 0
        for c in n1:
            if c in n2:
                common += 1
        
        return common / max(len(n1), len(n2))
    
    def generate_recommendations(self) -> List[Dict]:
        """Generate recommendations for dealing with duplicates"""
        recommendations = []
        
        # Backend service duplicates
        if "backend_services_video" in self.duplicates:
            recommendations.append({
                "category": "Backend Video Services",
                "issue": "Multiple video generation services detected",
                "files": [
                    "video_generation.py",
                    "video_generation_pipeline.py", 
                    "video_generation_orchestrator.py",
                    "enhanced_video_generation.py"
                ],
                "recommendation": "Consolidate into video_generation_pipeline.py as the main service, make others import from it",
                "priority": "HIGH"
            })
        
        if "backend_services_payment" in self.duplicates:
            recommendations.append({
                "category": "Payment Services",
                "issue": "Duplicate payment services",
                "files": ["payment_service.py", "payment_service_enhanced.py"],
                "recommendation": "Keep payment_service_enhanced.py, remove or deprecate payment_service.py",
                "priority": "MEDIUM"
            })
        
        if "backend_services_analytics" in self.duplicates:
            recommendations.append({
                "category": "Analytics Services",
                "issue": "Multiple analytics services",
                "files": ["analytics_service.py", "realtime_analytics_service.py"],
                "recommendation": "Merge realtime features into main analytics_service.py",
                "priority": "MEDIUM"
            })
        
        if "backend_services_cost" in self.duplicates:
            recommendations.append({
                "category": "Cost Tracking",
                "issue": "Duplicate cost tracking services",
                "files": ["cost_tracking.py", "realtime_cost_tracking.py"],
                "recommendation": "Merge into single cost_tracking.py with realtime capabilities",
                "priority": "MEDIUM"
            })
        
        # Check for missing critical features
        if self.task_status["week2"]["P0"]["percentage"] < 100:
            missing = [t["name"] for t in self.task_status["week2"]["P0"]["tasks"] if not t["completed"]]
            if missing:
                recommendations.append({
                    "category": "Missing Features",
                    "issue": "Critical Week 2 P0 features incomplete",
                    "files": missing,
                    "recommendation": "Complete implementation of missing features",
                    "priority": "CRITICAL"
                })
        
        return recommendations


def print_analysis_report(results: Dict):
    """Print formatted analysis report"""
    
    print("\n" + "="*80)
    print("ANALYSIS RESULTS")
    print("="*80)
    
    # Task Completion Summary
    print("\n[TASK COMPLETION STATUS]")
    print("-"*40)
    
    for week, priorities in results["task_completion"].items():
        print(f"\n{week.upper()}:")
        for priority, data in priorities.items():
            if isinstance(data, dict) and "percentage" in data:
                status = "[COMPLETE]" if data["percentage"] == 100 else "[INCOMPLETE]"
                print(f"  {priority}: {data['percentage']:.0f}% ({data['completed']}/{data['total']}) {status}")
    
    # Duplicates Found
    print("\n[DUPLICATE SERVICES & COMPONENTS]")
    print("-"*40)
    
    if results["duplicates"]:
        for category, items in results["duplicates"].items():
            print(f"\n{category.replace('_', ' ').title()}:")
            for item in items:
                if "files" in item:
                    print(f"  - Files: {', '.join(item['files'])}")
                    print(f"    Similarity: {item.get('similarity', 'N/A'):.1%}")
                    print(f"    Action: {item.get('recommendation', 'Review')}")
    else:
        print("No significant duplicates found")
    
    # Similar Files
    if results["similar_files"]:
        print("\n[SIMILAR FILES DETECTED]")
        print("-"*40)
        for category, items in results["similar_files"].items():
            if items:
                print(f"\n{category.replace('_', ' ').title()}:")
                for item in items[:5]:  # Show top 5
                    print(f"  - {item['file1']} <-> {item['file2']} ({item['similarity']:.1%})")
    
    # Recommendations
    print("\n[RECOMMENDATIONS]")
    print("-"*40)
    
    if results["recommendations"]:
        for i, rec in enumerate(results["recommendations"], 1):
            print(f"\n{i}. [{rec['priority']}] {rec['category']}")
            print(f"   Issue: {rec['issue']}")
            print(f"   Files: {', '.join(rec['files']) if isinstance(rec['files'], list) else rec['files']}")
            print(f"   Action: {rec['recommendation']}")
    else:
        print("No critical issues found")
    
    # Save results
    output_file = "project_duplicate_analysis.json"
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2, default=str)
    
    print(f"\n[REPORT SAVED]: {output_file}")


if __name__ == "__main__":
    analyzer = ProjectAnalyzer()
    results = analyzer.analyze_project()
    print_analysis_report(results)