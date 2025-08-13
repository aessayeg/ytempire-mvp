#!/usr/bin/env python3
"""
Service Integration Analysis Script
Analyzes all services and determines integration approach
"""

import os
import re
from typing import Dict, List, Tuple, Optional

def analyze_service_file(file_path: str) -> Dict:
    """Analyze a service file to determine integration requirements"""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # Check for class definitions
        classes = re.findall(r'class\s+(\w+)', content)
        
        # Check for initialize/shutdown methods
        has_initialize = bool(re.search(r'async def initialize', content))
        has_shutdown = bool(re.search(r'async def shutdown', content))
        
        # Check for global instances
        instances = re.findall(r'^([a-z_][a-z0-9_]*)\s*=\s*\w+\(\)', content, re.MULTILINE)
        
        # Check for imports and dependencies
        imports = re.findall(r'from\s+(\S+)\s+import|import\s+(\S+)', content)
        
        return {
            'file': os.path.basename(file_path),
            'size': len(content),
            'classes': classes,
            'has_initialize': has_initialize,
            'has_shutdown': has_shutdown,
            'instances': instances,
            'imports': [imp[0] or imp[1] for imp in imports],
            'lines': len(content.split('\n'))
        }
    except Exception as e:
        return {
            'file': os.path.basename(file_path),
            'error': str(e)
        }

def identify_duplicates(services: List[Dict]) -> Dict[str, List[Dict]]:
    """Identify potential duplicate services"""
    duplicates = {}
    
    # Group by similar names
    groups = {
        'analytics': ['analytics_service', 'analytics_pipeline', 'analytics_connector'],
        'cost': ['cost_tracking', 'cost_optimizer', 'cost_aggregation', 'cost_verification'],
        'gpu': ['gpu_resource_service', 'gpu_resource_manager'],
        'metrics': ['metrics_pipeline', 'metrics_pipeline_operational', 'metrics_aggregation'],
        'reporting': ['reporting', 'reporting_service', 'reporting_infrastructure', 'automated_reporting'],
        'video_gen': ['video_generation_orchestrator', 'video_generation_pipeline', 'video_processor'],
        'video_quick': ['mock_video_generator', 'quick_video_generator'],
        'vector_db': ['vector_database', 'vector_database_deployed'],
        'websocket': ['websocket_manager', 'websocket_events'],
        'youtube': ['youtube_service', 'youtube_multi_account', 'youtube_oauth_service']
    }
    
    for group_name, patterns in groups.items():
        group_services = []
        for service in services:
            if any(pattern in service['file'] for pattern in patterns):
                group_services.append(service)
        
        if len(group_services) > 1:
            duplicates[group_name] = group_services
    
    return duplicates

def main():
    services_dir = r"C:\Users\Hp\projects\ytempire-mvp\backend\app\services"
    services = []
    
    # Analyze all service files
    for file in os.listdir(services_dir):
        if file.endswith('.py') and not file.startswith('__'):
            file_path = os.path.join(services_dir, file)
            analysis = analyze_service_file(file_path)
            services.append(analysis)
    
    # Sort by size (larger = more complete)
    services.sort(key=lambda x: x.get('lines', 0), reverse=True)
    
    # Identify duplicates
    duplicates = identify_duplicates(services)
    
    print("=== SERVICE INTEGRATION ANALYSIS ===\n")
    
    # Currently integrated services
    integrated = [
        'realtime_analytics_service.py',
        'beta_success_metrics.py', 
        'scaling_optimizer.py',
        'cost_tracking.py',
        'gpu_resource_service.py',
        'youtube_multi_account.py',
        'alert_service.py'
    ]
    
    print("CURRENTLY INTEGRATED (7 services):")
    for service in integrated:
        print(f"  ✅ {service}")
    
    print(f"\nDUPLICATE GROUPS DETECTED ({len(duplicates)} groups):")
    for group_name, group_services in duplicates.items():
        print(f"\n{group_name.upper()}:")
        for service in group_services:
            marker = "✅" if service['file'] in integrated else "❌"
            print(f"  {marker} {service['file']} ({service.get('lines', 0)} lines, "
                  f"init: {'✅' if service.get('has_initialize') else '❌'}, "
                  f"instances: {len(service.get('instances', []))})")
    
    print(f"\nUNIQUE SERVICES TO INTEGRATE:")
    unique_services = []
    for service in services:
        if service['file'] not in integrated:
            is_duplicate = False
            for group_services in duplicates.values():
                if service in group_services:
                    # Check if this is the largest in its group
                    largest = max(group_services, key=lambda x: x.get('lines', 0))
                    if service != largest:
                        is_duplicate = True
                        break
            
            if not is_duplicate:
                unique_services.append(service)
    
    for service in unique_services:
        print(f"  🔄 {service['file']} ({service.get('lines', 0)} lines)")
    
    print(f"\nINTEGRATION SUMMARY:")
    print(f"  - Currently integrated: {len(integrated)} services")
    print(f"  - Duplicate groups: {len(duplicates)} groups") 
    print(f"  - Services to integrate: {len(unique_services)} services")
    print(f"  - Total integration target: {len(integrated) + len(unique_services)} services")

if __name__ == '__main__':
    main()