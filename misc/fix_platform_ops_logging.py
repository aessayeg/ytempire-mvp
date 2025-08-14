#!/usr/bin/env python3
"""
Fix Platform Ops P2 components logging paths for Windows compatibility
"""

import os
import re
from pathlib import Path
from typing import List

def fix_logging_paths():
    """Fix logging paths in Platform Ops P2 components"""
    
    project_root = Path("C:/Users/Hp/projects/ytempire-mvp")
    
    # Files to fix
    files_to_fix = [
        project_root / "infrastructure/orchestration/service_mesh_evaluation.py",
        project_root / "infrastructure/monitoring/advanced_dashboards.py",
        project_root / "infrastructure/testing/chaos_engineering_suite.py",
        project_root / "infrastructure/deployment/multi_region_deployment_planner.py"
    ]
    
    # Create logs directory
    logs_dir = project_root / "logs" / "platform_ops"
    logs_dir.mkdir(parents=True, exist_ok=True)
    print(f"Created logs directory: {logs_dir}")
    
    # Fix each file
    for file_path in files_to_fix:
        if file_path.exists():
            print(f"\nFixing {file_path.name}...")
            
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            original_content = content
            
            # Replace Unix-style log paths with Windows-compatible paths
            replacements = [
                (r"'/var/log/ytempire/([^']+)'", rf"'{logs_dir}/\1'".replace('\\', '/')),
                (r'"/var/log/ytempire/([^"]+)"', rf'"{logs_dir}/\1"'.replace('\\', '/')),
                (r"logging\.FileHandler\('/var/log/ytempire/([^']+)'\)", 
                 rf"logging.FileHandler('{logs_dir}/\1')".replace('\\', '/')),
                (r"logging\.FileHandler\(\"/var/log/ytempire/([^\"]+)\"\)", 
                 rf'logging.FileHandler("{logs_dir}/\1")'.replace('\\', '/'))
            ]
            
            for pattern, replacement in replacements:
                content = re.sub(pattern, replacement, content)
            
            # Also fix os.makedirs calls
            content = re.sub(
                r"os\.makedirs\(os\.path\.dirname\('/var/log/ytempire/[^']+'\), exist_ok=True\)",
                f"os.makedirs('{logs_dir}', exist_ok=True)".replace('\\', '/'),
                content
            )
            
            content = re.sub(
                r'os\.makedirs\(os\.path\.dirname\("/var/log/ytempire/[^"]+"\), exist_ok=True\)',
                f'os.makedirs("{logs_dir}", exist_ok=True)'.replace('\\', '/'),
                content
            )
            
            # Fix specific path references
            content = content.replace('/var/log/ytempire/', f'{logs_dir}/'.replace('\\', '/'))
            
            if content != original_content:
                with open(file_path, 'w', encoding='utf-8') as f:
                    f.write(content)
                print(f"  ✓ Fixed logging paths in {file_path.name}")
            else:
                print(f"  - No changes needed in {file_path.name}")
        else:
            print(f"  ✗ File not found: {file_path}")
    
    print(f"\n✓ Logging paths fixed for Windows compatibility")
    print(f"  Log files will be stored in: {logs_dir}")
    
    return logs_dir

def create_logging_config():
    """Create a centralized logging configuration for Platform Ops"""
    
    project_root = Path("C:/Users/Hp/projects/ytempire-mvp")
    logs_dir = project_root / "logs" / "platform_ops"
    
    config_content = f'''"""
Platform Ops Logging Configuration
Provides consistent logging setup across all Platform Ops components
"""

import logging
import os
from pathlib import Path

# Platform Ops log directory
PLATFORM_OPS_LOG_DIR = Path(r"{logs_dir}")
PLATFORM_OPS_LOG_DIR.mkdir(parents=True, exist_ok=True)

def get_logger(name: str, log_file: str = None) -> logging.Logger:
    """
    Get a configured logger for Platform Ops components
    
    Args:
        name: Logger name (usually __name__)
        log_file: Optional specific log file name
    
    Returns:
        Configured logger instance
    """
    logger = logging.getLogger(name)
    
    if not logger.handlers:
        # Set level
        logger.setLevel(logging.INFO)
        
        # Create formatter
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        
        # Console handler
        console_handler = logging.StreamHandler()
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)
        
        # File handler (optional)
        if log_file:
            file_path = PLATFORM_OPS_LOG_DIR / log_file
            file_handler = logging.FileHandler(file_path)
            file_handler.setFormatter(formatter)
            logger.addHandler(file_handler)
    
    return logger

# Convenience function for getting component-specific loggers
def get_service_mesh_logger():
    return get_logger('service_mesh_evaluation', 'service_mesh_evaluation.log')

def get_dashboard_logger():
    return get_logger('advanced_dashboards', 'dashboard_manager.log')

def get_chaos_logger():
    return get_logger('chaos_engineering', 'chaos_engineering.log')

def get_deployment_logger():
    return get_logger('multi_region_deployment', 'multi_region_deployment.log')
'''
    
    config_file = project_root / "infrastructure" / "platform_ops_logging.py"
    with open(config_file, 'w', encoding='utf-8') as f:
        f.write(config_content)
    
    print(f"\n✓ Created centralized logging configuration at: {config_file}")
    
    return config_file

def main():
    """Main execution"""
    print("=" * 60)
    print("Fixing Platform Ops P2 Logging Configuration")
    print("=" * 60)
    
    # Fix logging paths
    logs_dir = fix_logging_paths()
    
    # Create centralized config
    config_file = create_logging_config()
    
    print("\n" + "=" * 60)
    print("Logging Configuration Complete!")
    print("=" * 60)
    print(f"\n✓ Log directory: {logs_dir}")
    print(f"✓ Config file: {config_file}")
    print("\nAll Platform Ops components now use Windows-compatible paths.")
    
    return 0

if __name__ == "__main__":
    exit(main())