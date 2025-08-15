#!/usr/bin/env python3
"""
Script to fix common TypeScript/JSX syntax errors in frontend components
"""

import os
import re
from pathlib import Path
from typing import List, Tuple

def fix_jsx_semicolons(content: str) -> str:
    """Fix JSX expressions that incorrectly have semicolons"""
    # Pattern for JSX expressions with semicolons
    patterns = [
        # Remove semicolons before closing JSX braces
        (r';\s*}>', '}>', 'Remove semicolon before JSX closing brace'),
        (r';\s*}\)', '})', 'Remove semicolon before JSX closing with paren'),
        # Fix semicolons in JSX attributes
        (r'=\{([^}]+);\s*}', r'={\1}', 'Remove semicolon in JSX attribute'),
        # Fix semicolons in conditional renders
        (r'\?\s*\(([^)]+);\s*\)', r'? (\1)', 'Remove semicolon in ternary expression'),
        (r':\s*\(([^)]+);\s*\)', r': (\1)', 'Remove semicolon in ternary else'),
    ]
    
    for pattern, replacement, description in patterns:
        content = re.sub(pattern, replacement, content)
    
    return content

def fix_template_literals(content: str) -> str:
    """Fix malformed template literals"""
    # Fix unterminated template literals
    lines = content.split('\n')
    fixed_lines = []
    in_template = False
    template_start_line = -1
    
    for i, line in enumerate(lines):
        # Count backticks
        backticks = line.count('`')
        
        # Check for unterminated template literals
        if '`' in line and backticks % 2 != 0:
            if not in_template:
                in_template = True
                template_start_line = i
                fixed_lines.append(line)
            else:
                # Close the template literal
                if not line.rstrip().endswith('`'):
                    line = line.rstrip() + '`'
                in_template = False
                fixed_lines.append(line)
        else:
            fixed_lines.append(line)
    
    content = '\n'.join(fixed_lines)
    
    # Fix template literal syntax
    patterns = [
        # Fix spacing in template literals
        (r'\$\s+\{', '${', 'Fix spacing in template literal'),
        # Fix nested template literals
        (r'`([^`]*)\$\{([^}]+)}\s*;`', r'`\1${\2}`', 'Remove semicolon in template literal'),
    ]
    
    for pattern, replacement, description in patterns:
        content = re.sub(pattern, replacement, content)
    
    return content

def fix_ternary_operators(content: str) -> str:
    """Fix malformed ternary operators"""
    patterns = [
        # Fix ternary with missing spaces
        (r'(\w+)\?(\w+):(\w+)', r'\1 ? \2 : \3', 'Add spaces to ternary'),
        # Fix ternary with semicolons
        (r'\?\s*([^:]+);\s*:', r'? \1 :', 'Remove semicolon in ternary condition'),
        # Fix nested ternaries
        (r'(\?\s*[^:]+\s*:\s*[^?]+)\s*;(\s*\?)', r'\1\2', 'Remove semicolon between ternaries'),
    ]
    
    for pattern, replacement, description in patterns:
        content = re.sub(pattern, replacement, content)
    
    return content

def fix_object_syntax(content: str) -> str:
    """Fix object literal syntax errors"""
    patterns = [
        # Fix trailing commas in objects
        (r',\s*}', '}', 'Remove trailing comma in object'),
        # Fix missing commas between properties
        (r'(\w+:\s*[^,}\n]+)\n\s*(\w+:)', r'\1,\n  \2', 'Add missing comma between properties'),
        # Fix semicolons in objects
        (r'(\w+:\s*[^;}\n]+);\s*([,}])', r'\1\2', 'Replace semicolon with comma in object'),
    ]
    
    for pattern, replacement, description in patterns:
        content = re.sub(pattern, replacement, content, flags=re.MULTILINE)
    
    return content

def fix_jsx_expressions(content: str) -> str:
    """Fix JSX expression syntax"""
    patterns = [
        # Fix JSX expressions with multiple root elements
        (r'return\s*\(\s*<(\w+)', r'return (\n    <\1', 'Fix JSX return formatting'),
        # Fix JSX fragment syntax
        (r'<>\s*;', '<>', 'Remove semicolon after fragment'),
        (r'</>\s*;', '</>', 'Remove semicolon after closing fragment'),
        # Fix JSX attribute expressions
        (r'=\{([^}]+);\}', r'={\1}', 'Remove semicolon in JSX attribute'),
    ]
    
    for pattern, replacement, description in patterns:
        content = re.sub(pattern, replacement, content)
    
    return content

def fix_import_export_syntax(content: str) -> str:
    """Fix import/export statement syntax"""
    patterns = [
        # Fix double semicolons
        (r';;', ';', 'Remove double semicolons'),
        # Fix import statements
        (r'import\s+{([^}]+)}\s*from\s*(["\'][^"\']+["\']);?', r'import { \1 } from \2;', 'Fix import formatting'),
        # Fix export statements
        (r'export\s+default\s+(\w+);;', r'export default \1;', 'Fix export default'),
    ]
    
    for pattern, replacement, description in patterns:
        content = re.sub(pattern, replacement, content)
    
    return content

def process_file(file_path: Path) -> Tuple[bool, List[str]]:
    """Process a single file and fix syntax errors"""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        original_content = content
        fixes_applied = []
        
        # Apply all fixes
        content = fix_jsx_semicolons(content)
        if content != original_content:
            fixes_applied.append("Fixed JSX semicolons")
        
        temp = content
        content = fix_template_literals(content)
        if content != temp:
            fixes_applied.append("Fixed template literals")
        
        temp = content
        content = fix_ternary_operators(content)
        if content != temp:
            fixes_applied.append("Fixed ternary operators")
        
        temp = content
        content = fix_object_syntax(content)
        if content != temp:
            fixes_applied.append("Fixed object syntax")
        
        temp = content
        content = fix_jsx_expressions(content)
        if content != temp:
            fixes_applied.append("Fixed JSX expressions")
        
        temp = content
        content = fix_import_export_syntax(content)
        if content != temp:
            fixes_applied.append("Fixed import/export syntax")
        
        # Write back if changes were made
        if content != original_content:
            with open(file_path, 'w', encoding='utf-8') as f:
                f.write(content)
            return True, fixes_applied
        
        return False, []
    
    except Exception as e:
        return False, [f"Error: {str(e)}"]

def main():
    """Main function to fix syntax errors in all TypeScript/TSX files"""
    frontend_dir = Path("C:/Users/Hp/projects/ytempire-mvp/frontend/src")
    
    # Files with known issues
    priority_files = [
        "components/Analytics/AnalyticsDashboard.tsx",
        "components/Analytics/CompetitiveAnalysisDashboard.tsx",
        "components/BatchOperations/BatchOperations.tsx",
        "components/Accessibility/AccessibleButton.tsx",
        "components/Analytics/UserBehaviorDashboard.tsx",
        "components/Animations/AdvancedAnimations.tsx",
        "components/Animations/styledComponents.ts",
        "components/Animations/variants.ts",
        "components/Auth/EmailVerification.tsx",
        "components/Auth/ForgotPasswordForm.tsx",
        "components/Auth/LoginForm.tsx",
        "components/Auth/RegisterForm.tsx",
        "components/Auth/TwoFactorAuth.tsx",
        "components/BulkOperations/EnhancedBulkOperations.tsx",
        "components/ChannelManager/ChannelManager.tsx",
        "components/Channels/BulkOperations.tsx",
        "components/Channels/ChannelDashboard.tsx",
        "components/Channels/ChannelHealthDashboard.tsx",
        "components/Channels/ChannelList.tsx",
        "components/Channels/ChannelTemplates.tsx",
        "components/Charts/AdvancedCharts.tsx",
        "components/Charts/ChannelPerformanceCharts.tsx",
        "components/Charts/ChartComponents.tsx",
        "components/Charts/index.tsx",
        "components/Common/ErrorMessage.tsx",
        "components/Common/InlineHelp.tsx",
        "components/CostTracking/CostVisualization.tsx",
        "components/Dashboard/BusinessIntelligenceDashboard.tsx",
        "components/Dashboard/CustomizableWidgets.tsx",
        "components/Dashboard/DashboardLayout.tsx",
        "components/Dashboard/EnhancedMetricsDashboard.tsx",
        "components/Dashboard/MainDashboard.tsx",
    ]
    
    print("Fixing TypeScript/JSX Syntax Errors")
    print("=" * 50)
    
    total_fixed = 0
    errors = []
    
    # Process priority files first
    for file_rel_path in priority_files:
        file_path = frontend_dir / file_rel_path
        if file_path.exists():
            fixed, fixes = process_file(file_path)
            if fixed:
                total_fixed += 1
                print(f"Fixed {file_path.name}: {', '.join(fixes)}")
            elif fixes:
                errors.append(f"{file_path.name}: {fixes[0]}")
    
    # Process all other TypeScript/TSX files
    for file_path in frontend_dir.rglob("*.tsx"):
        rel_path = file_path.relative_to(frontend_dir)
        if str(rel_path).replace('\\', '/') not in priority_files:
            fixed, fixes = process_file(file_path)
            if fixed:
                total_fixed += 1
                print(f"Fixed {file_path.name}: {', '.join(fixes)}")
    
    for file_path in frontend_dir.rglob("*.ts"):
        if not str(file_path).endswith('.d.ts'):
            rel_path = file_path.relative_to(frontend_dir)
            if str(rel_path).replace('\\', '/') not in priority_files:
                fixed, fixes = process_file(file_path)
                if fixed:
                    total_fixed += 1
                    print(f"Fixed {file_path.name}: {', '.join(fixes)}")
    
    print("\n" + "=" * 50)
    print(f"Summary: Fixed {total_fixed} files")
    
    if errors:
        print(f"\nErrors encountered in {len(errors)} files:")
        for error in errors[:10]:  # Show first 10 errors
            print(f"  - {error}")
    
    return total_fixed > 0

if __name__ == "__main__":
    main()