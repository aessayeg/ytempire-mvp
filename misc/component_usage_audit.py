#!/usr/bin/env python3
"""
Component Usage Audit Script for YTEmpire MVP Frontend

This script analyzes all React components in the frontend/src/components directory
to identify orphaned/unused components that are not imported anywhere else in the codebase.
"""

import os
import re
from pathlib import Path
from typing import Dict, List, Set, Tuple
import json

class ComponentAuditor:
    def __init__(self, frontend_path: str):
        self.frontend_path = Path(frontend_path)
        self.src_path = self.frontend_path / "src"
        self.components_path = self.src_path / "components"
        
    def get_all_components(self) -> Dict[str, str]:
        """Get all component files and their paths."""
        components = {}
        
        for root, dirs, files in os.walk(self.components_path):
            for file in files:
                if file.endswith(('.tsx', '.ts', '.jsx', '.js')):
                    file_path = Path(root) / file
                    rel_path = file_path.relative_to(self.src_path)
                    component_name = file.replace('.tsx', '').replace('.ts', '').replace('.jsx', '').replace('.js', '')
                    components[str(rel_path)] = component_name
                    
        return components
    
    def extract_component_exports(self, file_path: Path) -> List[str]:
        """Extract exported component names from a file."""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
                
            exports = []
            
            # Default export patterns
            default_export_patterns = [
                r'export\s+default\s+(?:function\s+)?(\w+)',
                r'export\s+default\s+(\w+)',
                r'export\s+{\s*(\w+)\s+as\s+default\s*}',
            ]
            
            # Named export patterns
            named_export_patterns = [
                r'export\s+(?:const|let|var|function|class)\s+(\w+)',
                r'export\s+{\s*([^}]+)\s*}',
            ]
            
            # Find default exports
            for pattern in default_export_patterns:
                matches = re.findall(pattern, content, re.MULTILINE)
                exports.extend(matches)
            
            # Find named exports
            for pattern in named_export_patterns:
                matches = re.findall(pattern, content, re.MULTILINE)
                for match in matches:
                    if '{' not in match:  # Simple export
                        exports.append(match)
                    else:  # Multiple exports in braces
                        # Extract individual exports from braces
                        exports_in_braces = re.findall(r'(\w+)(?:\s+as\s+\w+)?', match)
                        exports.extend(exports_in_braces)
            
            return list(set(exports))  # Remove duplicates
            
        except Exception as e:
            print(f"Error reading {file_path}: {e}")
            return []
    
    def search_for_usage(self, component_name: str, export_names: List[str]) -> Dict[str, List[str]]:
        """Search for component usage across the entire src directory."""
        usage = {'imports': [], 'references': []}
        
        # Search patterns
        search_terms = [component_name] + export_names
        
        for root, dirs, files in os.walk(self.src_path):
            for file in files:
                if file.endswith(('.tsx', '.ts', '.jsx', '.js')):
                    file_path = Path(root) / file
                    
                    # Skip the component's own file
                    if file_path.name == f"{component_name}.tsx" or file_path.name == f"{component_name}.ts":
                        continue
                    
                    try:
                        with open(file_path, 'r', encoding='utf-8') as f:
                            content = f.read()
                            
                        # Check for imports
                        for term in search_terms:
                            import_patterns = [
                                rf'import\s+.*{re.escape(term)}.*from\s+[\'"][^\'"]*{re.escape(component_name)}[\'"]',
                                rf'import\s+.*{re.escape(term)}.*from\s+[\'"][^\'"]*components[^\'"]*[\'"]',
                                rf'import\s+{{\s*[^}}]*{re.escape(term)}[^}}]*\s*}}\s+from',
                                rf'import\s+{re.escape(term)}\s+from',
                            ]
                            
                            for pattern in import_patterns:
                                if re.search(pattern, content, re.MULTILINE | re.IGNORECASE):
                                    rel_path = file_path.relative_to(self.src_path)
                                    if str(rel_path) not in usage['imports']:
                                        usage['imports'].append(str(rel_path))
                            
                            # Check for usage in JSX/TSX
                            jsx_patterns = [
                                rf'<{re.escape(term)}\s*[/>]',
                                rf'<{re.escape(term)}\s+[^>]*>',
                                rf'{re.escape(term)}\s*\(',  # Function calls
                            ]
                            
                            for pattern in jsx_patterns:
                                if re.search(pattern, content):
                                    rel_path = file_path.relative_to(self.src_path)
                                    if str(rel_path) not in usage['references']:
                                        usage['references'].append(str(rel_path))
                                    
                    except Exception as e:
                        print(f"Error searching in {file_path}: {e}")
        
        return usage
    
    def audit_components(self) -> Dict:
        """Perform complete component audit."""
        components = self.get_all_components()
        results = {
            'total_components': len(components),
            'orphaned_components': [],
            'used_components': [],
            'component_details': {}
        }
        
        print(f"Found {len(components)} components to audit...")
        
        for comp_path, comp_name in components.items():
            print(f"Auditing: {comp_path}")
            
            full_path = self.src_path / comp_path
            exports = self.extract_component_exports(full_path)
            usage = self.search_for_usage(comp_name, exports)
            
            component_info = {
                'path': comp_path,
                'name': comp_name,
                'exports': exports,
                'imports': usage['imports'],
                'references': usage['references'],
                'total_usages': len(usage['imports']) + len(usage['references'])
            }
            
            results['component_details'][comp_path] = component_info
            
            # Classify as orphaned or used
            if component_info['total_usages'] == 0:
                results['orphaned_components'].append(component_info)
            else:
                results['used_components'].append(component_info)
        
        return results
    
    def generate_report(self, results: Dict) -> str:
        """Generate a formatted report of the audit results."""
        report = []
        report.append("# Frontend Component Usage Audit Report")
        report.append("=" * 50)
        report.append(f"Total Components Analyzed: {results['total_components']}")
        report.append(f"Orphaned Components: {len(results['orphaned_components'])}")
        report.append(f"Used Components: {len(results['used_components'])}")
        report.append("")
        
        # Orphaned Components Section
        report.append("## ORPHANED COMPONENTS (Not imported/used anywhere)")
        report.append("-" * 60)
        
        if results['orphaned_components']:
            # Group by directory
            orphaned_by_dir = {}
            for comp in results['orphaned_components']:
                dir_name = str(Path(comp['path']).parent)
                if dir_name not in orphaned_by_dir:
                    orphaned_by_dir[dir_name] = []
                orphaned_by_dir[dir_name].append(comp)
            
            for dir_name, comps in sorted(orphaned_by_dir.items()):
                report.append(f"\n### {dir_name}/")
                for comp in comps:
                    report.append(f"- **{comp['name']}** ({comp['path']})")
                    if comp['exports']:
                        report.append(f"  - Exports: {', '.join(comp['exports'])}")
        else:
            report.append("✅ No orphaned components found!")
        
        # Used Components Summary
        report.append("\n\n## USED COMPONENTS SUMMARY")
        report.append("-" * 40)
        
        # Group used components by directory
        used_by_dir = {}
        for comp in results['used_components']:
            dir_name = str(Path(comp['path']).parent)
            if dir_name not in used_by_dir:
                used_by_dir[dir_name] = []
            used_by_dir[dir_name].append(comp)
        
        for dir_name, comps in sorted(used_by_dir.items()):
            report.append(f"\n### {dir_name}/ ({len(comps)} components)")
            for comp in sorted(comps, key=lambda x: x['total_usages'], reverse=True):
                report.append(f"- **{comp['name']}** - Used in {comp['total_usages']} places")
        
        # Most/Least Used Components
        if results['used_components']:
            sorted_used = sorted(results['used_components'], key=lambda x: x['total_usages'], reverse=True)
            
            report.append("\n\n## MOST USED COMPONENTS (Top 10)")
            report.append("-" * 40)
            for comp in sorted_used[:10]:
                report.append(f"- **{comp['name']}** ({comp['total_usages']} usages)")
                report.append(f"  - Path: {comp['path']}")
                if comp['imports']:
                    report.append(f"  - Imported by: {', '.join(comp['imports'][:3])}{'...' if len(comp['imports']) > 3 else ''}")
        
        return "\n".join(report)

def main():
    frontend_path = r"C:\Users\Hp\projects\ytempire-mvp\frontend"
    
    auditor = ComponentAuditor(frontend_path)
    print("Starting component usage audit...")
    
    results = auditor.audit_components()
    
    # Generate report
    report = auditor.generate_report(results)
    
    # Save results
    report_file = Path(r"C:\Users\Hp\projects\ytempire-mvp\misc\component_audit_report.md")
    with open(report_file, 'w', encoding='utf-8') as f:
        f.write(report)
    
    results_file = Path(r"C:\Users\Hp\projects\ytempire-mvp\misc\component_audit_results.json")
    with open(results_file, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2)
    
    print(f"\n✅ Audit complete!")
    print(f"Report saved to: {report_file}")
    print(f"Raw results saved to: {results_file}")
    
    # Print summary
    print(f"\n📊 SUMMARY:")
    print(f"Total components: {results['total_components']}")
    print(f"Orphaned components: {len(results['orphaned_components'])}")
    print(f"Used components: {len(results['used_components'])}")
    
    if results['orphaned_components']:
        print(f"\n🚨 ORPHANED COMPONENTS FOUND:")
        for comp in results['orphaned_components'][:10]:  # Show first 10
            print(f"- {comp['path']}")
        if len(results['orphaned_components']) > 10:
            print(f"... and {len(results['orphaned_components']) - 10} more")

if __name__ == "__main__":
    main()