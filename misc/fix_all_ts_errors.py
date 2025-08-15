#!/usr/bin/env python3
"""
Comprehensive script to fix all TypeScript syntax errors in the frontend
"""

import os
import re
from pathlib import Path

def fix_ts_file(file_path):
    """Fix TypeScript syntax errors in a single file"""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        original_content = content
        
        # Fix unterminated template literals
        content = re.sub(r'`\s*$', '', content, flags=re.MULTILINE)
        content = re.sub(r'`\s*\n', '\n', content, flags=re.MULTILINE)
        
        # Fix arrow function syntax errors
        content = re.sub(r'forEach\(_\(([\w]+)\)\s*=>', r'forEach((\1) =>', content)
        content = re.sub(r'map\(_\(([\w]+)\)\s*=>', r'map((\1) =>', content)
        content = re.sub(r'filter\(_\(([\w]+)\)\s*=>', r'filter((\1) =>', content)
        content = re.sub(r'MutationObserver\(_\(([\w]+)\)\s*=>', r'MutationObserver((\1) =>', content)
        
        # Fix catch blocks
        content = re.sub(r'catch\s*\(_\)\s*\{', r'catch (error) {', content)
        
        # Fix malformed object syntax
        content = re.sub(r'\}\)\}(?!\s*[;,\)])', r'});\n}', content)
        content = re.sub(r'\}\)\}\s*else', r'});\n    } else', content)
        
        # Fix missing semicolons in specific patterns
        content = re.sub(r'(\w+\s*:\s*[\w\s]+),\s*\}', r'\1\n}', content)
        
        # Fix "in, window" to "in window"
        content = re.sub(r'in,\s*window', r'in window', content)
        
        # Fix property assignment issues in object literals
        lines = content.split('\n')
        fixed_lines = []
        in_object = False
        brace_count = 0
        
        for i, line in enumerate(lines):
            # Track if we're inside an object literal
            brace_count += line.count('{') - line.count('}')
            
            # Fix lines that look like incomplete property assignments
            if re.match(r'^\s+\w+:\s*$', line):
                # This is likely an incomplete property, skip it
                continue
            
            # Fix lines with trailing commas before closing braces
            line = re.sub(r',\s*\}', '\n}', line)
            
            fixed_lines.append(line)
        
        content = '\n'.join(fixed_lines)
        
        # Only write if content changed
        if content != original_content:
            with open(file_path, 'w', encoding='utf-8') as f:
                f.write(content)
            return True
        return False
    except Exception as e:
        print(f"Error processing {file_path}: {e}")
        return False

def find_ts_tsx_files(directory):
    """Find all TypeScript and TSX files"""
    ts_files = []
    for root, dirs, files in os.walk(directory):
        # Skip node_modules and build directories
        dirs[:] = [d for d in dirs if d not in ['node_modules', 'dist', 'build', '.git']]
        
        for file in files:
            if file.endswith(('.ts', '.tsx')):
                ts_files.append(os.path.join(root, file))
    
    return ts_files

def main():
    frontend_dir = r"C:\Users\Hp\projects\ytempire-mvp\frontend"
    
    print("Finding TypeScript files...")
    ts_files = find_ts_tsx_files(frontend_dir)
    print(f"Found {len(ts_files)} TypeScript files")
    
    fixed_count = 0
    for file_path in ts_files:
        if fix_ts_file(file_path):
            fixed_count += 1
            print(f"Fixed: {os.path.relpath(file_path, frontend_dir)}")
    
    print(f"\nFixed {fixed_count} files")
    print("Run 'npm run build' to verify all errors are resolved")

if __name__ == "__main__":
    main()