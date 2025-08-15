#!/usr/bin/env python3
"""
Script to fix common TypeScript syntax errors in the frontend codebase
"""

import re
import os
from pathlib import Path

def fix_file_content(content):
    """Fix common TypeScript syntax errors in file content"""
    
    # Fix unterminated template literals - look for stray backticks
    content = re.sub(r'`\s*$', '', content, flags=re.MULTILINE)
    content = re.sub(r'`\s*\n\s*$', '', content, flags=re.MULTILINE)
    
    # Fix arrow function syntax errors (_(param) => should be (param) =>)
    content = re.sub(r'forEach\(_\(([\w]+)\)\s*=>', r'forEach((\1) =>', content)
    content = re.sub(r'map\(_\(([\w]+)\)\s*=>', r'map((\1) =>', content)
    content = re.sub(r'filter\(_\(([\w]+)\)\s*=>', r'filter((\1) =>', content)
    
    # Fix catch blocks with underscore
    content = re.sub(r'catch\s*\(_\)\s*\{', r'catch (error) {', content)
    
    # Fix variable names with spaces (e.g., "h1 Elements" -> "h1Elements")
    content = re.sub(r'const\s+(\w+)\s+(\w+)\s*=', r'const \1\2 =', content)
    content = re.sub(r'let\s+(\w+)\s+(\w+)\s*=', r'let \1\2 =', content)
    
    # Fix object/interface declarations with incorrect commas
    content = re.sub(r',\s*}', r'\n}', content)
    
    # Fix missing semicolons at end of statements
    lines = content.split('\n')
    fixed_lines = []
    for i, line in enumerate(lines):
        # Check if line ends with ) or } and next line starts with a keyword
        if line.strip() and not line.strip().endswith((';', '{', '}', ',', ':')) and \
           not line.strip().startswith('//') and \
           re.search(r'\)\s*$', line):
            # Check if it's not part of an if/while/for statement
            if not re.search(r'^\s*(if|while|for|catch|switch)\s*\(', line):
                line = line.rstrip() + ';'
        fixed_lines.append(line)
    content = '\n'.join(fixed_lines)
    
    return content

def fix_typescript_files(directory):
    """Fix TypeScript files in the given directory"""
    frontend_path = Path(directory)
    
    # Files that need fixing based on error output
    files_to_fix = [
        'src/components/Accessibility/announcementManager.ts',
        'src/components/Analytics/AnalyticsDashboard.tsx',
        'src/theme/darkMode.ts',
        'src/theme/materialTheme.ts',
        'src/utils/accessibility.ts',
        'src/utils/EventEmitter.ts',
        'src/utils/formatters.ts',
    ]
    
    for file_path in files_to_fix:
        full_path = frontend_path / file_path
        if full_path.exists():
            print(f"Fixing {file_path}...")
            try:
                with open(full_path, 'r', encoding='utf-8') as f:
                    content = f.read()
                
                # Create backup
                backup_path = full_path.with_suffix('.backup')
                with open(backup_path, 'w', encoding='utf-8') as f:
                    f.write(content)
                
                # Fix content
                fixed_content = fix_file_content(content)
                
                # Write fixed content
                with open(full_path, 'w', encoding='utf-8') as f:
                    f.write(fixed_content)
                
                print(f"  [OK] Fixed {file_path}")
            except Exception as e:
                print(f"  [ERROR] Error fixing {file_path}: {e}")

if __name__ == "__main__":
    frontend_dir = r"C:\Users\Hp\projects\ytempire-mvp\frontend"
    fix_typescript_files(frontend_dir)
    print("\nDone! Now run 'npm run build' to check if errors are resolved.")