"""
Service Dependency Validation
Maps and validates all service dependencies to prevent circular dependencies and orphaned services
"""

import sys
import os
import ast
import json
from typing import Dict, List, Set, Tuple, Any, Optional
from datetime import datetime
import networkx as nx
import matplotlib.pyplot as plt
from collections import defaultdict

# Add project paths
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'backend'))

class ServiceDependencyValidator:
    """Validate service dependencies and integration"""
    
    def __init__(self):
        self.services = {}
        self.dependencies = defaultdict(set)
        self.circular_deps = []
        self.orphaned_services = []
        self.missing_imports = []
        self.results = {
            "services": {},
            "dependencies": {},
            "issues": {},
            "health": {}
        }
        
    def scan_services(self) -> Dict[str, Any]:
        """Scan all services in the backend"""
        print("\n📂 Scanning Services Directory...")
        
        services_path = os.path.join(os.path.dirname(__file__), '..', 'backend', 'app', 'services')
        
        if not os.path.exists(services_path):
            self.results["services"]["error"] = "Services directory not found"
            return {}
            
        service_files = [f for f in os.listdir(services_path) 
                        if f.endswith('.py') and not f.startswith('__')]
        
        print(f"  Found {len(service_files)} service files")
        
        for service_file in service_files:
            service_name = service_file[:-3]  # Remove .py extension
            file_path = os.path.join(services_path, service_file)
            
            # Analyze imports
            imports = self.analyze_imports(file_path)
            self.services[service_name] = {
                "file": service_file,
                "path": file_path,
                "imports": imports,
                "dependencies": set(),
                "dependents": set()
            }
            
        self.results["services"] = {
            "total": len(service_files),
            "scanned": len(self.services),
            "list": list(self.services.keys())[:10]  # Show first 10
        }
        
        return self.services
        
    def analyze_imports(self, file_path: str) -> List[str]:
        """Analyze imports in a Python file"""
        imports = []
        
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
                
            # Parse AST to find imports
            tree = ast.parse(content)
            
            for node in ast.walk(tree):
                if isinstance(node, ast.Import):
                    for alias in node.names:
                        imports.append(alias.name)
                elif isinstance(node, ast.ImportFrom):
                    if node.module:
                        imports.append(node.module)
                        
        except Exception as e:
            print(f"    Error analyzing {file_path}: {e}")
            
        return imports
        
    def build_dependency_graph(self) -> nx.DiGraph:
        """Build dependency graph from services"""
        print("\n🔗 Building Dependency Graph...")
        
        graph = nx.DiGraph()
        
        # Add nodes
        for service in self.services:
            graph.add_node(service)
            
        # Add edges based on imports
        for service_name, service_info in self.services.items():
            for imp in service_info["imports"]:
                # Check if import is another service
                if "app.services." in imp:
                    dep_service = imp.split("app.services.")[-1].split(".")[0]
                    if dep_service in self.services and dep_service != service_name:
                        graph.add_edge(service_name, dep_service)
                        self.services[service_name]["dependencies"].add(dep_service)
                        self.services[dep_service]["dependents"].add(service_name)
                        
        self.results["dependencies"]["total_edges"] = graph.number_of_edges()
        self.results["dependencies"]["graph_density"] = nx.density(graph)
        
        return graph
        
    def detect_circular_dependencies(self, graph: nx.DiGraph) -> List[List[str]]:
        """Detect circular dependencies in the graph"""
        print("\n🔄 Detecting Circular Dependencies...")
        
        try:
            cycles = list(nx.simple_cycles(graph))
            
            if cycles:
                print(f"  ⚠️ Found {len(cycles)} circular dependencies!")
                for cycle in cycles:
                    cycle.append(cycle[0])  # Complete the cycle for display
                    print(f"    Cycle: {' -> '.join(cycle)}")
                    self.circular_deps.append(cycle)
            else:
                print("  ✅ No circular dependencies detected")
                
            self.results["issues"]["circular_dependencies"] = {
                "count": len(cycles),
                "cycles": cycles[:5] if cycles else []  # Show first 5
            }
            
        except Exception as e:
            print(f"  Error detecting cycles: {e}")
            self.results["issues"]["circular_dependencies"] = {
                "error": str(e)
            }
            
        return self.circular_deps
        
    def find_orphaned_services(self, graph: nx.DiGraph) -> List[str]:
        """Find services with no dependencies or dependents"""
        print("\n🏝️ Finding Orphaned Services...")
        
        orphaned = []
        
        for service in self.services:
            in_degree = graph.in_degree(service)
            out_degree = graph.out_degree(service)
            
            # Service is orphaned if it has no connections
            if in_degree == 0 and out_degree == 0:
                orphaned.append(service)
                
        if orphaned:
            print(f"  ⚠️ Found {len(orphaned)} orphaned services:")
            for service in orphaned[:10]:  # Show first 10
                print(f"    - {service}")
        else:
            print("  ✅ No orphaned services found")
            
        self.orphaned_services = orphaned
        self.results["issues"]["orphaned_services"] = {
            "count": len(orphaned),
            "list": orphaned[:10]  # Show first 10
        }
        
        return orphaned
        
    def validate_critical_services(self) -> Dict[str, Any]:
        """Validate critical service dependencies"""
        print("\n⚡ Validating Critical Services...")
        
        critical_services = {
            "video_generation_pipeline": ["cost_tracking", "ai_services", "youtube_service"],
            "cost_tracking": [],  # Should have minimal dependencies
            "authentication": [],  # Should be independent
            "websocket_manager": ["notification_service"],
            "payment_service_enhanced": ["subscription_service"],
            "ml_integration_service": ["ai_services", "cost_tracking"],
            "realtime_analytics_service": ["websocket_manager"],
            "batch_processing": ["notification_service", "cost_tracking"]
        }
        
        validation_results = {}
        
        for service, expected_deps in critical_services.items():
            if service in self.services:
                actual_deps = self.services[service]["dependencies"]
                
                # Check if expected dependencies are present
                missing = set(expected_deps) - actual_deps
                extra = actual_deps - set(expected_deps)
                
                validation_results[service] = {
                    "status": "✅ Valid" if not missing else "⚠️ Missing dependencies",
                    "expected": expected_deps,
                    "actual": list(actual_deps),
                    "missing": list(missing),
                    "extra": list(extra)
                }
                
                if not missing:
                    print(f"  ✅ {service}: All critical dependencies present")
                else:
                    print(f"  ⚠️ {service}: Missing {missing}")
            else:
                validation_results[service] = {
                    "status": "❌ Service not found"
                }
                print(f"  ❌ {service}: Service not found")
                
        self.results["critical_services"] = validation_results
        return validation_results
        
    def check_service_health(self) -> Dict[str, Any]:
        """Check health endpoints for services"""
        print("\n🏥 Checking Service Health Patterns...")
        
        health_patterns = {
            "health_check": 0,
            "status_method": 0,
            "ping_endpoint": 0,
            "error_handling": 0,
            "logging": 0,
            "monitoring": 0
        }
        
        for service_name, service_info in self.services.items():
            try:
                with open(service_info["path"], 'r', encoding='utf-8') as f:
                    content = f.read().lower()
                    
                if "health" in content or "status" in content:
                    health_patterns["health_check"] += 1
                if "def status" in content or "def get_status" in content:
                    health_patterns["status_method"] += 1
                if "ping" in content:
                    health_patterns["ping_endpoint"] += 1
                if "try:" in content and "except" in content:
                    health_patterns["error_handling"] += 1
                if "logger" in content or "logging" in content:
                    health_patterns["logging"] += 1
                if "monitor" in content or "metric" in content:
                    health_patterns["monitoring"] += 1
                    
            except Exception as e:
                pass
                
        total_services = len(self.services)
        health_score = {
            pattern: f"{(count/total_services*100):.1f}%" 
            for pattern, count in health_patterns.items()
        }
        
        self.results["health"] = {
            "patterns": health_patterns,
            "coverage": health_score,
            "overall_score": f"{(sum(health_patterns.values())/(len(health_patterns)*total_services)*100):.1f}%"
        }
        
        print(f"  Overall health coverage: {self.results['health']['overall_score']}")
        
        return health_score
        
    def analyze_service_layers(self, graph: nx.DiGraph) -> Dict[str, List[str]]:
        """Analyze service layers and architecture"""
        print("\n🏗️ Analyzing Service Architecture...")
        
        # Categorize services by layer
        layers = {
            "core": [],
            "business": [],
            "integration": [],
            "utility": [],
            "data": []
        }
        
        for service in self.services:
            if any(keyword in service for keyword in ["auth", "security", "core"]):
                layers["core"].append(service)
            elif any(keyword in service for keyword in ["video", "channel", "payment", "subscription"]):
                layers["business"].append(service)
            elif any(keyword in service for keyword in ["youtube", "ai", "ml", "webhook", "notification"]):
                layers["integration"].append(service)
            elif any(keyword in service for keyword in ["cache", "queue", "batch", "websocket"]):
                layers["utility"].append(service)
            elif any(keyword in service for keyword in ["data", "analytics", "feature", "training"]):
                layers["data"].append(service)
            else:
                # Try to categorize based on dependencies
                deps = self.services[service]["dependencies"]
                if len(deps) == 0:
                    layers["core"].append(service)
                elif len(deps) > 5:
                    layers["integration"].append(service)
                else:
                    layers["business"].append(service)
                    
        # Analyze layer violations (lower layers depending on higher layers)
        layer_order = ["core", "utility", "data", "business", "integration"]
        violations = []
        
        for i, layer in enumerate(layer_order):
            for service in layers[layer]:
                if service in self.services:
                    deps = self.services[service]["dependencies"]
                    for dep in deps:
                        # Find which layer the dependency is in
                        for j, check_layer in enumerate(layer_order):
                            if dep in layers[check_layer]:
                                if j > i:  # Dependency is in a higher layer
                                    violations.append({
                                        "service": service,
                                        "layer": layer,
                                        "depends_on": dep,
                                        "dep_layer": check_layer
                                    })
                                    
        self.results["architecture"] = {
            "layers": {k: len(v) for k, v in layers.items()},
            "violations": len(violations),
            "violation_examples": violations[:5]  # Show first 5
        }
        
        if violations:
            print(f"  ⚠️ Found {len(violations)} layer violations")
        else:
            print("  ✅ Clean layered architecture")
            
        return layers
        
    def generate_dependency_visualization(self, graph: nx.DiGraph):
        """Generate visual representation of dependencies"""
        print("\n📊 Generating Dependency Visualization...")
        
        try:
            # Create a smaller subgraph for visualization (top services)
            important_services = [
                "video_generation_pipeline",
                "cost_tracking",
                "ml_integration_service",
                "websocket_manager",
                "batch_processing",
                "realtime_analytics_service",
                "payment_service_enhanced",
                "youtube_multi_account",
                "ai_services",
                "notification_service"
            ]
            
            subgraph_nodes = []
            for service in important_services:
                if service in graph:
                    subgraph_nodes.append(service)
                    # Add direct dependencies
                    subgraph_nodes.extend(list(graph.successors(service))[:2])
                    subgraph_nodes.extend(list(graph.predecessors(service))[:2])
                    
            subgraph = graph.subgraph(subgraph_nodes)
            
            # Calculate metrics
            if len(subgraph.nodes) > 0:
                avg_degree = sum(dict(subgraph.degree()).values()) / len(subgraph.nodes)
                max_in = max(dict(subgraph.in_degree()).values()) if subgraph.nodes else 0
                max_out = max(dict(subgraph.out_degree()).values()) if subgraph.nodes else 0
                
                self.results["graph_metrics"] = {
                    "nodes": len(graph.nodes),
                    "edges": len(graph.edges),
                    "avg_connections": f"{avg_degree:.2f}",
                    "max_dependencies": max_out,
                    "max_dependents": max_in
                }
            else:
                self.results["graph_metrics"] = {
                    "nodes": 0,
                    "edges": 0,
                    "error": "No nodes in subgraph"
                }
                
            print("  ✅ Visualization data prepared")
            
        except Exception as e:
            print(f"  ❌ Error generating visualization: {e}")
            self.results["graph_metrics"] = {"error": str(e)}
            
    def generate_report(self) -> str:
        """Generate dependency validation report"""
        report = []
        report.append("=" * 80)
        report.append("SERVICE DEPENDENCY VALIDATION REPORT")
        report.append("=" * 80)
        report.append(f"Validation Date: {datetime.now().isoformat()}")
        report.append("")
        
        # Services Summary
        report.append("\n📂 SERVICES SUMMARY:")
        report.append("-" * 40)
        if "services" in self.results:
            report.append(f"  Total Services: {self.results['services'].get('total', 0)}")
            report.append(f"  Scanned: {self.results['services'].get('scanned', 0)}")
            
        # Dependency Analysis
        report.append("\n🔗 DEPENDENCY ANALYSIS:")
        report.append("-" * 40)
        if "dependencies" in self.results:
            report.append(f"  Total Dependencies: {self.results['dependencies'].get('total_edges', 0)}")
            report.append(f"  Graph Density: {self.results['dependencies'].get('graph_density', 0):.3f}")
            
        # Issues
        report.append("\n⚠️ ISSUES DETECTED:")
        report.append("-" * 40)
        if "issues" in self.results:
            if "circular_dependencies" in self.results["issues"]:
                circ_count = self.results["issues"]["circular_dependencies"].get("count", 0)
                report.append(f"  Circular Dependencies: {circ_count}")
                if circ_count > 0:
                    report.append("    Examples:")
                    for cycle in self.results["issues"]["circular_dependencies"].get("cycles", [])[:3]:
                        report.append(f"      {' -> '.join(cycle)}")
                        
            if "orphaned_services" in self.results["issues"]:
                orph_count = self.results["issues"]["orphaned_services"].get("count", 0)
                report.append(f"  Orphaned Services: {orph_count}")
                if orph_count > 0:
                    report.append("    Examples:")
                    for service in self.results["issues"]["orphaned_services"].get("list", [])[:5]:
                        report.append(f"      - {service}")
                        
        # Critical Services
        report.append("\n⚡ CRITICAL SERVICES:")
        report.append("-" * 40)
        if "critical_services" in self.results:
            for service, validation in self.results["critical_services"].items():
                if isinstance(validation, dict):
                    status = validation.get("status", "Unknown")
                    report.append(f"  {service}: {status}")
                    
        # Architecture
        report.append("\n🏗️ ARCHITECTURE ANALYSIS:")
        report.append("-" * 40)
        if "architecture" in self.results:
            report.append("  Layer Distribution:")
            for layer, count in self.results["architecture"].get("layers", {}).items():
                report.append(f"    {layer}: {count} services")
            violations = self.results["architecture"].get("violations", 0)
            report.append(f"  Layer Violations: {violations}")
            
        # Health Coverage
        report.append("\n🏥 HEALTH COVERAGE:")
        report.append("-" * 40)
        if "health" in self.results:
            report.append(f"  Overall Score: {self.results['health'].get('overall_score', 'N/A')}")
            for pattern, coverage in self.results["health"].get("coverage", {}).items():
                report.append(f"    {pattern}: {coverage}")
                
        # Graph Metrics
        report.append("\n📊 GRAPH METRICS:")
        report.append("-" * 40)
        if "graph_metrics" in self.results:
            for metric, value in self.results["graph_metrics"].items():
                report.append(f"  {metric}: {value}")
                
        # Summary
        report.append("\n" + "=" * 80)
        report.append("SUMMARY")
        report.append("=" * 80)
        
        # Calculate health score
        issues_count = 0
        if "issues" in self.results:
            issues_count += self.results["issues"].get("circular_dependencies", {}).get("count", 0)
            issues_count += min(5, self.results["issues"].get("orphaned_services", {}).get("count", 0))
            
        health_status = "✅ HEALTHY" if issues_count == 0 else "⚠️ NEEDS ATTENTION" if issues_count < 5 else "❌ CRITICAL ISSUES"
        
        report.append(f"Overall Status: {health_status}")
        report.append(f"Total Issues: {issues_count}")
        
        return "\n".join(report)
        
    def run_validation(self) -> bool:
        """Run complete service dependency validation"""
        print("Starting Service Dependency Validation...")
        print("=" * 80)
        
        # Scan services
        self.scan_services()
        
        # Build dependency graph
        graph = self.build_dependency_graph()
        
        # Run validations
        self.detect_circular_dependencies(graph)
        self.find_orphaned_services(graph)
        self.validate_critical_services()
        self.check_service_health()
        self.analyze_service_layers(graph)
        self.generate_dependency_visualization(graph)
        
        # Generate report
        report = self.generate_report()
        print("\n" + report)
        
        # Save report
        report_file = os.path.join(os.path.dirname(__file__), 'service_dependency_validation_report.txt')
        with open(report_file, 'w') as f:
            f.write(report)
        print(f"\nReport saved to: {report_file}")
        
        # Save dependency map as JSON
        dep_map = {
            service: {
                "dependencies": list(info["dependencies"]),
                "dependents": list(info["dependents"])
            }
            for service, info in self.services.items()
        }
        
        dep_map_file = os.path.join(os.path.dirname(__file__), 'service_dependency_map.json')
        with open(dep_map_file, 'w') as f:
            json.dump(dep_map, f, indent=2)
        print(f"Dependency map saved to: {dep_map_file}")
        
        # Determine success
        critical_issues = len(self.circular_deps) + min(10, len(self.orphaned_services))
        return critical_issues < 5

if __name__ == "__main__":
    validator = ServiceDependencyValidator()
    success = validator.run_validation()
    
    if success:
        print("\n✅ Service Dependency Validation PASSED!")
        print("No critical dependency issues found.")
        sys.exit(0)
    else:
        print("\n⚠️ Service Dependency Validation found issues.")
        print("Review the report for recommendations.")
        sys.exit(1)