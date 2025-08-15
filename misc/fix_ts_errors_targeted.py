#!/usr/bin/env python3
"""
Targeted script to fix specific TypeScript syntax errors.
"""

import os
import re
from pathlib import Path

def fix_object_syntax_errors(content: str) -> str:
    """Fix object property syntax errors like {, and },"""
    # Fix pattern: {, at start of object properties
    content = re.sub(r'(\w+\s*:\s*)\{,(\s)', r'\1{\2', content)
    
    # Fix pattern: }, appearing alone  
    content = re.sub(r'^(\s*)\},(\s*)$', r'\1}\2', content, flags=re.MULTILINE)
    
    return content

def fix_hex_colors(content: str) -> str:
    """Fix hex color codes with spaces"""
    # Pattern for 6-digit hex with spaces
    content = re.sub(r'#([0-9a-fA-F]{1,2})\s+([0-9a-fA-F]{1,2})\s+([0-9a-fA-F]{1,2})', r'#\1\2\3', content)
    
    # Pattern for other hex formats with spaces
    content = re.sub(r'#([0-9a-fA-F]{2})\s+([0-9a-fA-F]{2})\s+([0-9a-fA-F]{2})', r'#\1\2\3', content)
    content = re.sub(r'#([0-9a-fA-F])\s+([0-9a-fA-F])\s+([0-9a-fA-F])', r'#\1\2\3', content)
    
    return content

def fix_css_units(content: str) -> str:
    """Fix CSS units with spaces"""
    # Fix rem units
    content = re.sub(r'(\d+(?:\.\d+)?)\s+rem', r'\1rem', content)
    
    # Fix linear-gradient deg
    content = re.sub(r'linear-gradient\((\d+)\s+deg', r'linear-gradient(\1deg', content)
    
    return content

def fix_switch_statements(content: str) -> str:
    """Add missing default cases to switch statements"""
    lines = content.split('\n')
    fixed_lines = []
    in_switch = False
    switch_indent = 0
    has_default = False
    last_case_indent = 0
    
    for i, line in enumerate(lines):
        if 'switch' in line and '{' in line:
            in_switch = True
            switch_indent = len(line) - len(line.lstrip())
            has_default = False
            fixed_lines.append(line)
        elif in_switch:
            if 'case ' in line:
                last_case_indent = len(line) - len(line.lstrip())
            if 'default:' in line:
                has_default = True
            
            # Check if we're at the closing brace of the switch
            if '}' in line and len(line) - len(line.lstrip()) == switch_indent + 2:
                if not has_default:
                    # Add default case before closing brace
                    default_line = ' ' * (last_case_indent) + 'default:'
                    return_line = ' ' * (last_case_indent + 2) + 'return e.key.toLowerCase() === key;'
                    fixed_lines.append(default_line)
                    fixed_lines.append(return_line)
                in_switch = False
            
            fixed_lines.append(line)
        else:
            fixed_lines.append(line)
    
    return '\n'.join(fixed_lines)

def fix_template_literals(content: str) -> str:
    """Fix unterminated template literals"""
    lines = content.split('\n')
    fixed_lines = []
    
    for i, line in enumerate(lines):
        # Count backticks in the line
        backtick_count = line.count('`')
        
        # If odd number of backticks, likely unterminated
        if backtick_count % 2 == 1:
            # Check if line ends with an open parenthesis or similar
            if '${' in line and ')' not in line:
                # Template literal with expression, likely missing closing backtick
                if not line.rstrip().endswith('`'):
                    line = line.rstrip() + '`'
        
        fixed_lines.append(line)
    
    return '\n'.join(fixed_lines)

def fix_file(file_path: Path) -> list[str]:
    """Fix a single TypeScript file"""
    fixes = []
    
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        original = content
        
        # Apply fixes
        content = fix_object_syntax_errors(content)
        if content != original:
            fixes.append("Fixed object syntax")
            original = content
        
        content = fix_hex_colors(content)
        if content != original:
            fixes.append("Fixed hex colors")
            original = content
        
        content = fix_css_units(content)
        if content != original:
            fixes.append("Fixed CSS units")
            original = content
        
        # Only fix switch statements in specific files
        if 'AccessibleButton' in str(file_path):
            content = fix_switch_statements(content)
            if content != original:
                fixes.append("Fixed switch statement")
                original = content
        
        content = fix_template_literals(content)
        if content != original:
            fixes.append("Fixed template literals")
        
        if fixes:
            with open(file_path, 'w', encoding='utf-8') as f:
                f.write(content)
    
    except Exception as e:
        print(f"[ERROR] {file_path}: {e}")
    
    return fixes

def main():
    # List of files with known errors to fix
    files_to_fix = [
        r"C:\Users\Hp\projects\ytempire-mvp\frontend\src\theme\accessibleTheme.ts",
        r"C:\Users\Hp\projects\ytempire-mvp\frontend\src\components\Analytics\AnalyticsDashboard.tsx",
        r"C:\Users\Hp\projects\ytempire-mvp\frontend\src\stores\channelStore.ts",
        r"C:\Users\Hp\projects\ytempire-mvp\frontend\src\stores\useChannelStore.ts",
        r"C:\Users\Hp\projects\ytempire-mvp\frontend\src\stores\videoStore.ts",
    ]
    
    total_fixed = 0
    
    for file_path in files_to_fix:
        path = Path(file_path)
        if path.exists():
            fixes = fix_file(path)
            if fixes:
                print(f"[FIXED] {path.name}: {', '.join(fixes)}")
                total_fixed += 1
        else:
            print(f"[SKIP] {path.name}: File not found")
    
    print(f"\n{'='*60}")
    print(f"Fixed {total_fixed} files")
    print(f"{'='*60}")

if __name__ == "__main__":
    main()