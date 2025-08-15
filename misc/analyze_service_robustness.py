"""
Analyze service robustness to identify which files to keep
"""

import os
import re
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Tuple
import json

class RobustnessAnalyzer:
    def __init__(self):
        self.services_path = Path("backend/app/services")
        self.results = {}
        
    def analyze_file_robustness(self, file_path: Path) -> Dict:
        """Analyze a file's robustness based on multiple factors"""
        try:
            with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                content = f.read()
            
            # Metrics for robustness
            metrics = {
                "file_name": file_path.name,
                "size": file_path.stat().st_size,
                "lines": len(content.splitlines()),
                "classes": len(re.findall(r'^class\s+\w+', content, re.MULTILINE)),
                "functions": len(re.findall(r'^def\s+\w+', content, re.MULTILINE)),
                "async_functions": len(re.findall(r'^async\s+def\s+\w+', content, re.MULTILINE)),
                "imports": len(re.findall(r'^import\s+|^from\s+', content, re.MULTILINE)),
                "error_handling": len(re.findall(r'try:|except\s+\w+|except:', content)),
                "logging": len(re.findall(r'logger\.|logging\.', content)),
                "type_hints": len(re.findall(r'->\s+\w+|:\s+\w+\[|:\s+Dict|:\s+List|:\s+Optional', content)),
                "docstrings": len(re.findall(r'"""[\s\S]*?"""', content)),
                "tests": len(re.findall(r'test_|assert\s+|unittest|pytest', content)),
                "celery_tasks": len(re.findall(r'@celery_app\.task|@app\.task', content)),
                "database_ops": len(re.findall(r'db\.session|AsyncSession|select\(|insert\(|update\(', content)),
                "api_integration": len(re.findall(r'requests\.|aiohttp|httpx|api_key|endpoint', content)),
                "has_singleton": 'Singleton' in content or '= \w+\(\)$' in content,
                "completeness_score": 0
            }
            
            # Calculate completeness score
            score = 0
            score += min(30, metrics["functions"] * 2)  # Functions (max 30)
            score += min(20, metrics["classes"] * 5)    # Classes (max 20)
            score += min(15, metrics["error_handling"] * 3)  # Error handling (max 15)
            score += min(10, metrics["logging"] * 2)    # Logging (max 10)
            score += min(10, metrics["type_hints"])     # Type hints (max 10)
            score += min(10, metrics["docstrings"] * 2) # Documentation (max 10)
            score += 5 if metrics["has_singleton"] else 0  # Singleton pattern
            
            metrics["completeness_score"] = score
            
            return metrics
            
        except Exception as e:
            print(f"Error analyzing {file_path}: {e}")
            return {}
    
    def analyze_service_category(self, category: str, files: List[str]) -> Dict:
        """Analyze all files in a category and rank them"""
        category_analysis = []
        
        for file_name in files:
            file_path = self.services_path / f"{file_name}.py"
            if file_path.exists():
                metrics = self.analyze_file_robustness(file_path)
                if metrics:
                    category_analysis.append(metrics)
        
        # Sort by completeness score
        category_analysis.sort(key=lambda x: x["completeness_score"], reverse=True)
        
        return {
            "files": category_analysis,
            "best_file": category_analysis[0] if category_analysis else None,
            "recommendation": self.generate_recommendation(category_analysis)
        }
    
    def generate_recommendation(self, files: List[Dict]) -> Dict:
        """Generate consolidation recommendation"""
        if not files:
            return {}
        
        best = files[0]
        
        # Identify files to merge (high-quality secondary files)
        merge_candidates = [
            f for f in files[1:] 
            if f["completeness_score"] > 50 and f["functions"] > 5
        ]
        
        # Identify files to delete (low quality or test files)
        delete_candidates = [
            f for f in files[1:]
            if "mock" in f["file_name"].lower() or 
               "test" in f["file_name"].lower() or
               "quick" in f["file_name"].lower() or
               f["completeness_score"] < 30
        ]
        
        return {
            "keep": best["file_name"],
            "merge": [f["file_name"] for f in merge_candidates],
            "delete": [f["file_name"] for f in delete_candidates],
            "reason": f"Best file has score {best['completeness_score']} with {best['functions']} functions and {best['classes']} classes"
        }
    
    def analyze_all_categories(self):
        """Analyze all service categories"""
        
        # Define categories with their files
        categories = {
            "video_generation": [
                "video_generation_pipeline",
                "video_generation_orchestrator", 
                "enhanced_video_generation",
                "video_processor",
                "video_queue_service",
                "video_pipeline",
                "mock_video_generator",
                "quick_video_generator",
                "analytics_pipeline",
                "etl_pipeline_service",
                "inference_pipeline",
                "metrics_pipeline",
                "metrics_pipeline_operational",
                "training_pipeline_service"
            ],
            "analytics": [
                "analytics_service",
                "realtime_analytics_service",
                "analytics_pipeline",
                "analytics_connector",
                "analytics_report",
                "automated_reporting",
                "beta_success_metrics",
                "custom_report_builder",
                "metrics_aggregation",
                "quality_metrics",
                "reporting",
                "reporting_infrastructure",
                "reporting_service",
                "user_behavior_analytics"
            ],
            "cost_tracking": [
                "cost_tracking",
                "realtime_cost_tracking",
                "cost_aggregation",
                "cost_optimizer",
                "cost_verification",
                "revenue_tracking",
                "defect_tracking"
            ],
            "payment": [
                "payment_service_enhanced",
                "subscription_service",
                "invoice_generator"
            ]
        }
        
        print("\n" + "="*100)
        print("SERVICE ROBUSTNESS ANALYSIS")
        print("="*100)
        
        recommendations = {}
        
        for category, files in categories.items():
            print(f"\n[{category.upper()}]")
            analysis = self.analyze_service_category(category, files)
            
            if analysis["best_file"]:
                best = analysis["best_file"]
                print(f"  Best Implementation: {best['file_name']}")
                print(f"    Score: {best['completeness_score']}/100")
                print(f"    Size: {best['size']:,} bytes")
                print(f"    Functions: {best['functions']}, Classes: {best['classes']}")
                print(f"    Error Handling: {best['error_handling']}, Logging: {best['logging']}")
                
                rec = analysis["recommendation"]
                if rec:
                    print(f"\n  Recommendation:")
                    print(f"    KEEP: {rec['keep']}")
                    if rec['merge']:
                        print(f"    MERGE: {', '.join(rec['merge'])}")
                    if rec['delete']:
                        print(f"    DELETE: {', '.join(rec['delete'])}")
                    
                    recommendations[category] = rec
        
        self.results = recommendations
        return recommendations
    
    def check_import_dependencies(self):
        """Check which files import the services we plan to delete"""
        print("\n" + "="*100)
        print("IMPORT DEPENDENCY ANALYSIS")
        print("="*100)
        
        # Files we plan to delete
        files_to_delete = set()
        for cat, rec in self.results.items():
            files_to_delete.update(rec.get("delete", []))
            files_to_delete.update(rec.get("merge", []))
        
        # Check all Python files for imports
        all_py_files = list(Path("backend").rglob("*.py"))
        
        import_map = {}
        
        for file_to_delete in files_to_delete:
            import_map[file_to_delete] = []
            
            for py_file in all_py_files:
                try:
                    with open(py_file, 'r', encoding='utf-8', errors='ignore') as f:
                        content = f.read()
                    
                    # Check for imports
                    if file_to_delete.replace(".py", "") in content:
                        if "from app.services." + file_to_delete.replace(".py", "") in content or \
                           "import " + file_to_delete.replace(".py", "") in content:
                            import_map[file_to_delete].append(str(py_file.relative_to(Path("backend"))))
                except:
                    pass
        
        # Print dependencies
        print("\nFiles that import services to be deleted/merged:")
        for service, importers in import_map.items():
            if importers:
                print(f"\n  {service}:")
                for imp in importers[:5]:  # Show first 5
                    print(f"    - {imp}")
                if len(importers) > 5:
                    print(f"    ... and {len(importers) - 5} more")
        
        return import_map


def main():
    analyzer = RobustnessAnalyzer()
    
    # Analyze all categories
    recommendations = analyzer.analyze_all_categories()
    
    # Check import dependencies
    import_dependencies = analyzer.check_import_dependencies()
    
    # Save results
    results = {
        "timestamp": datetime.now().isoformat(),
        "recommendations": recommendations,
        "import_dependencies": {
            k: v for k, v in import_dependencies.items() if v
        }
    }
    
    with open("robustness_analysis.json", "w") as f:
        json.dump(results, f, indent=2)
    
    print("\n[ANALYSIS COMPLETE] Results saved to robustness_analysis.json")
    
    return results


if __name__ == "__main__":
    results = main()