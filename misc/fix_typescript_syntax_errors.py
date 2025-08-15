#!/usr/bin/env python3
"""
Script to fix common TypeScript syntax errors in the frontend codebase.
"""

import os
import re
from pathlib import Path
import json

def fix_file_content(content: str, file_path: str) -> tuple[str, list[str]]:
    """Fix common TypeScript syntax errors in file content."""
    original_content = content
    fixes_made = []
    
    # Fix pattern 1: Object property with comma after opening brace
    # Pattern: {, or {,\n
    pattern1 = re.compile(r'(\{),(\s*[\n\r])')
    if pattern1.search(content):
        content = pattern1.sub(r'\1\2', content)
        fixes_made.append("Fixed object syntax: removed comma after opening brace")
    
    # Fix pattern 2: Space in hex color codes
    # Pattern: #XXX XXX or #XX XX XX
    pattern2 = re.compile(r'#([0-9a-fA-F]{1,2})\s+([0-9a-fA-F]{1,2})\s+([0-9a-fA-F]{1,2})')
    if pattern2.search(content):
        content = pattern2.sub(r'#\1\2\3', content)
        fixes_made.append("Fixed hex color codes: removed spaces")
    
    pattern2b = re.compile(r'#([0-9a-fA-F]{3,4})\s+([0-9a-fA-F]{3})')
    if pattern2b.search(content):
        content = pattern2b.sub(r'#\1\2', content)
        fixes_made.append("Fixed hex color codes: removed spaces")
    
    # Fix pattern 3: Space in rem units
    # Pattern: X.X rem -> X.Xrem
    pattern3 = re.compile(r'(\d+(?:\.\d+)?)\s+rem')
    if pattern3.search(content):
        content = pattern3.sub(r'\1rem', content)
        fixes_made.append("Fixed rem units: removed spaces")
    
    # Fix pattern 4: linear-gradient with space before deg
    # Pattern: linear-gradient(XXX deg -> linear-gradient(XXXdeg
    pattern4 = re.compile(r'linear-gradient\((\d+)\s+deg')
    if pattern4.search(content):
        content = pattern4.sub(r'linear-gradient(\1deg', content)
        fixes_made.append("Fixed linear-gradient: removed space before deg")
    
    # Fix pattern 5: Missing closing brace for objects
    # Pattern: },\n at wrong indentation
    pattern5 = re.compile(r'^(\s*)([a-zA-Z_$][a-zA-Z0-9_$]*)\s*:\s*\{,\s*$', re.MULTILINE)
    if pattern5.search(content):
        content = pattern5.sub(r'\1\2: {', content)
        fixes_made.append("Fixed object property syntax: removed comma after opening brace")
    
    # Fix pattern 6: Fix unterminated template literals
    # Look for template literals that start but don't end on the same or next line
    lines = content.split('\n')
    fixed_lines = []
    i = 0
    while i < len(lines):
        line = lines[i]
        # Check if line has opening backtick but no closing backtick
        if '`' in line and line.count('`') % 2 == 1:
            # Check if it's missing the closing backtick
            if line.strip().endswith('(') or line.strip().endswith(','):
                # Likely missing closing backtick
                if '${' in line:
                    # Has template expression, find the end
                    if i + 1 < len(lines) and ':' in lines[i + 1]:
                        # Next line continues, add closing backtick
                        line = line.rstrip() + '`'
                        fixes_made.append(f"Fixed unterminated template literal on line {i+1}")
            elif not line.strip().endswith('`'):
                # Check if next line should be part of template
                if i + 1 < len(lines):
                    next_line = lines[i + 1]
                    if next_line.strip().startswith(':'):
                        # This is continuation, close template before colon
                        line = line.rstrip() + '`'
                        fixes_made.append(f"Fixed unterminated template literal on line {i+1}")
        fixed_lines.append(line)
        i += 1
    
    if fixes_made and 'template literal' in ' '.join(fixes_made):
        content = '\n'.join(fixed_lines)
    
    # Fix pattern 7: Missing default case in switch statements
    # This is more complex, need to find switch statements without default
    pattern7 = re.compile(r'(switch\s*\([^)]+\)\s*\{[^}]*?)(\n\s*)(return\s+[^}]+?)\}', re.DOTALL)
    def add_default_case(match):
        switch_body = match.group(1)
        indent = match.group(2)
        last_return = match.group(3)
        if 'default:' not in switch_body:
            return f"{switch_body}{indent}default:{indent}  {last_return}}}"
        return match.group(0)
    
    if pattern7.search(content) and 'default:' not in content:
        content = pattern7.sub(add_default_case, content)
        fixes_made.append("Added missing default case to switch statement")
    
    # Fix pattern 8: Property assignment errors (extra commas in wrong places)
    # Pattern: main: '#xxx', light: '#yyy', should have commas between properties
    pattern8 = re.compile(r'(\w+):\s*([\'"][^\'"]+[\'"])\s*,\s*(\w+):\s*([\'"][^\'"]+[\'"])')
    
    # Fix pattern 9: Missing closing parentheses and braces
    # Count opening and closing braces/parens
    open_braces = content.count('{')
    close_braces = content.count('}')
    if open_braces > close_braces:
        # Add missing closing braces at the end
        content += '\n' + '}' * (open_braces - close_braces)
        fixes_made.append(f"Added {open_braces - close_braces} missing closing braces")
    
    open_parens = content.count('(')
    close_parens = content.count(')')
    if open_parens > close_parens:
        # Add missing closing parentheses
        content += ')' * (open_parens - close_parens)
        fixes_made.append(f"Added {open_parens - close_parens} missing closing parentheses")
    
    return content if content != original_content else (original_content, []), fixes_made


def process_directory(directory: Path):
    """Process all TypeScript/TSX files in the directory."""
    total_files = 0
    fixed_files = 0
    all_fixes = {}
    
    # Find all .ts and .tsx files
    ts_files = list(directory.rglob("*.ts"))
    tsx_files = list(directory.rglob("*.tsx"))
    all_files = ts_files + tsx_files
    
    # Skip node_modules and build directories
    all_files = [f for f in all_files if 'node_modules' not in str(f) and 'dist' not in str(f)]
    
    print(f"Found {len(all_files)} TypeScript files to check")
    
    for file_path in all_files:
        total_files += 1
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            fixed_content, fixes = fix_file_content(content, str(file_path))
            
            if fixes:
                # Write the fixed content back
                with open(file_path, 'w', encoding='utf-8') as f:
                    f.write(fixed_content)
                
                fixed_files += 1
                all_fixes[str(file_path)] = fixes
                print(f"[FIXED] {file_path.relative_to(directory)}: {', '.join(fixes)}")
        
        except Exception as e:
            print(f"[ERROR] Error processing {file_path}: {e}")
    
    # Save report
    report = {
        "total_files_processed": total_files,
        "files_fixed": fixed_files,
        "fixes_by_file": all_fixes
    }
    
    report_path = directory.parent / 'misc' / 'typescript_fixes_report.json'
    with open(report_path, 'w') as f:
        json.dump(report, f, indent=2)
    
    print(f"\n{'='*60}")
    print(f"TypeScript Syntax Fix Summary:")
    print(f"  Total files processed: {total_files}")
    print(f"  Files fixed: {fixed_files}")
    print(f"  Report saved to: {report_path}")
    print(f"{'='*60}")
    
    return fixed_files > 0


if __name__ == "__main__":
    frontend_dir = Path(r"C:\Users\Hp\projects\ytempire-mvp\frontend\src")
    
    if not frontend_dir.exists():
        print(f"Error: Directory {frontend_dir} does not exist")
        exit(1)
    
    print(f"Processing TypeScript files in: {frontend_dir}")
    print("=" * 60)
    
    success = process_directory(frontend_dir)
    
    if success:
        print("\n[SUCCESS] Successfully fixed TypeScript syntax errors!")
        print("Run 'npm run build' in the frontend directory to verify the fixes.")
    else:
        print("\n[SUCCESS] No syntax errors found to fix.")