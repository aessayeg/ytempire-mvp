#!/usr/bin/env python3
"""
Comprehensive fix for remaining TypeScript/JSX syntax errors
"""

import os
import re
import json
from pathlib import Path
from typing import List, Tuple, Dict

def analyze_lint_errors() -> Dict[str, List[str]]:
    """Run ESLint and parse errors"""
    import subprocess
    
    os.chdir("C:/Users/Hp/projects/ytempire-mvp/frontend")
    result = subprocess.run(["npm", "run", "lint"], capture_output=True, text=True)
    
    errors = {}
    for line in result.stdout.split('\n'):
        if 'Parsing error:' in line:
            parts = line.split()
            if len(parts) >= 3:
                file_part = parts[0]
                if '\\' in file_part:
                    file_name = file_part.split('\\')[-1]
                    if file_name not in errors:
                        errors[file_name] = []
                    errors[file_name].append(line)
    
    return errors

def fix_common_patterns(content: str) -> str:
    """Apply common pattern fixes"""
    
    # Fix property assignment errors
    content = re.sub(r'(\w+):\s*,', r'\1: null,', content)  # Empty property values
    content = re.sub(r'(\w+):\s*}', r'\1: null}', content)  # Last property without value
    
    # Fix object destructuring
    content = re.sub(r'\{\s*(\w+)\s*:\s*}', r'{ \1 }', content)  # Empty destructuring
    
    # Fix JSX issues
    content = re.sub(r'>\s*;\s*<', r'><', content)  # Remove semicolon between JSX elements
    content = re.sub(r'}\s*;\s*{', r'}{', content)  # Remove semicolon between JSX expressions
    
    # Fix comma issues in objects/arrays
    lines = content.split('\n')
    fixed_lines = []
    
    for i, line in enumerate(lines):
        # Add missing commas in object literals
        if re.match(r'^\s*\w+:\s*[^,}\n]+$', line) and i + 1 < len(lines):
            next_line = lines[i + 1] if i + 1 < len(lines) else ''
            if re.match(r'^\s*\w+:', next_line) or re.match(r'^\s*}', next_line):
                if not line.rstrip().endswith(','):
                    line = line.rstrip() + ',\n'
        
        # Fix interface/type declarations
        if 'interface' in line or 'type' in line:
            # Ensure semicolon before interface/type
            if i > 0 and lines[i-1].rstrip().endswith('}'):
                lines[i-1] = lines[i-1].rstrip() + ';\n'
        
        fixed_lines.append(line)
    
    return '\n'.join(fixed_lines)

def fix_jsx_fragments(content: str) -> str:
    """Fix JSX fragment issues"""
    
    # Find return statements with multiple elements
    pattern = r'return\s*\(\s*\n(\s*)<(\w+[^>]*>.*?</\w+>)\s*\n\s*<(\w+)'
    
    def replace_func(match):
        indent = match.group(1)
        first_elem = match.group(2)
        second_elem_start = match.group(3)
        return f'return (\n{indent}<>\n{indent}  <{first_elem}\n{indent}  <{second_elem_start}'
    
    content = re.sub(pattern, replace_func, content, flags=re.DOTALL)
    
    # Ensure fragments are properly closed
    if content.count('<>') > content.count('</>'):
        # Find unclosed fragments
        lines = content.split('\n')
        fragment_depth = 0
        for i, line in enumerate(lines):
            fragment_depth += line.count('<>') - line.count('</>')
            if fragment_depth > 0 and ')' in line and ';' in line:
                # Likely end of return statement
                lines[i] = line.replace(')', '</>\n  )')
                fragment_depth -= 1
        content = '\n'.join(lines)
    
    return content

def fix_specific_file_errors(file_path: Path) -> bool:
    """Fix errors in a specific file based on common patterns"""
    
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        original = content
        
        # Apply all fixes
        content = fix_common_patterns(content)
        content = fix_jsx_fragments(content)
        
        # File-specific fixes based on filename
        if 'styledComponents' in file_path.name:
            # Fix styled component syntax
            content = re.sub(r'styled\.\w+`([^`]*);`', r'styled.\w+`\1`', content)
            content = re.sub(r'css`([^`]*);`', r'css`\1`', content)
        
        if 'variants' in file_path.name:
            # Fix animation variant syntax
            content = re.sub(r'(\w+:\s*{[^}]*})\s*(\w+:)', r'\1,\n  \2', content)
        
        if 'Dashboard' in file_path.name:
            # Fix dashboard component patterns
            content = re.sub(r'data=\[\s*{', r'data={[{', content)
            content = re.sub(r'}\s*\]\s*>', r'}]}>', content)
        
        if content != original:
            with open(file_path, 'w', encoding='utf-8') as f:
                f.write(content)
            return True
        
        return False
        
    except Exception as e:
        print(f"Error processing {file_path.name}: {e}")
        return False

def main():
    """Main function to fix remaining syntax errors"""
    
    print("Analyzing and fixing remaining syntax errors...")
    print("=" * 50)
    
    frontend_src = Path("C:/Users/Hp/projects/ytempire-mvp/frontend/src")
    
    # Get all TypeScript/TSX files
    all_files = list(frontend_src.rglob("*.tsx")) + list(frontend_src.rglob("*.ts"))
    all_files = [f for f in all_files if not f.name.endswith('.d.ts')]
    
    fixed_count = 0
    
    for file_path in all_files:
        if fix_specific_file_errors(file_path):
            fixed_count += 1
            print(f"Fixed: {file_path.name}")
    
    print("=" * 50)
    print(f"Total files fixed: {fixed_count}")
    
    # Analyze remaining errors
    print("\nAnalyzing remaining errors...")
    try:
        errors = analyze_lint_errors()
        if errors:
            print(f"Files with remaining errors: {len(errors)}")
            for file_name, error_list in list(errors.items())[:5]:
                print(f"  - {file_name}: {len(error_list)} errors")
        else:
            print("No parsing errors found!")
    except Exception as e:
        print(f"Could not analyze errors: {e}")

if __name__ == "__main__":
    main()