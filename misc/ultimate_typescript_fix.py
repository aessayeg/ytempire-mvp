#!/usr/bin/env python3
"""
Ultimate TypeScript error fixer - 100% comprehensive
"""

import re
import os
from pathlib import Path

def fix_object_properties_in_context(content):
    """
    Fix object property separators based on context:
    - Interfaces/types use semicolons
    - Object literals use commas
    - Style props use commas
    """
    
    # Step 1: Fix object literals (use commas)
    # Match object assignments and array elements
    patterns = [
        # Object in return statement
        (r'return\s*\{([^}]+)\}', True),
        # Object in array
        (r'\[\s*\{([^}]+)\}', True),
        # Object assignment
        (r'=\s*\{([^}]+)\}', True),
        # Function argument object
        (r'\(\s*\{([^}]+)\}', True),
        # JSX prop object
        (r'=\{\{([^}]+)\}\}', True),
    ]
    
    for pattern, use_commas in patterns:
        def fix_props(match):
            content = match.group(1) if len(match.groups()) > 0 else match.group(0)
            if use_commas:
                # In object literals, use commas
                fixed = re.sub(r';(\s*[\w"\'])', r',\1', content)
                fixed = re.sub(r';(\s*\})', r'\1', fixed)
                fixed = re.sub(r';(\s*$)', '', fixed)
            else:
                # In interfaces/types, use semicolons
                fixed = re.sub(r',(\s*[\w"\'])', r';\1', content)
                fixed = re.sub(r',(\s*\})', r'\1', fixed)
            
            if len(match.groups()) > 0:
                return match.group(0).replace(match.group(1), fixed)
            return fixed
        
        content = re.sub(pattern, fix_props, content, flags=re.DOTALL)
    
    return content

def fix_interfaces_and_types(content):
    """Fix interface and type definitions to use semicolons"""
    
    # Fix interfaces
    def fix_interface(match):
        name = match.group(1)
        body = match.group(2)
        # Use semicolons in interfaces
        fixed_body = re.sub(r',(\s*\n\s*\w+\s*:)', r';\1', body)
        fixed_body = re.sub(r',(\s*\})', r'\1', fixed_body)
        return f'interface {name} {{{fixed_body}}}'
    
    content = re.sub(
        r'interface\s+(\w+)\s*\{([^}]*)\}',
        fix_interface,
        content,
        flags=re.DOTALL
    )
    
    # Fix type definitions
    def fix_type(match):
        name = match.group(1)
        body = match.group(2)
        # Use semicolons in types
        fixed_body = re.sub(r',(\s*\n\s*\w+\s*:)', r';\1', body)
        fixed_body = re.sub(r',(\s*\})', r'\1', fixed_body)
        return f'type {name} = {{{fixed_body}}}'
    
    content = re.sub(
        r'type\s+(\w+)\s*=\s*\{([^}]*)\}',
        fix_type,
        content,
        flags=re.DOTALL
    )
    
    return content

def fix_specific_patterns(content):
    """Fix specific TypeScript patterns that cause errors"""
    
    # Fix array/object literal syntax in specific contexts
    replacements = [
        # Fix semicolons in array literals
        (r'\[\s*\{([^}]*);([^}]*)\}', r'[{\1,\2}'),
        # Fix semicolons in object returns
        (r'return\s+\{([^}]*);([^}]*)\}', r'return {\1,\2}'),
        # Fix semicolons in JSX props
        (r'sx=\{\{([^}]*);([^}]*)\}\}', r'sx={{\1,\2}}'),
        (r'style=\{\{([^}]*);([^}]*)\}\}', r'style={{\1,\2}}'),
        # Fix template literals
        (r'``+', '`'),
        # Fix keyframes syntax
        (r'(keyframes)\s+([^`])', r'\1`\2'),
        (r'(styled\([^)]+\))\s+([^`])', r'\1`\2'),
        # Fix object syntax issues
        (r'\{,\s*', '{\n  '),
        (r',\s*\}', '\n}'),
        # Fix arrow functions
        (r'\)\s*=>\s*\}', r') => {}'),
        # Remove "No newline at end of file"
        (r'\s*No newline at end of file\s*', '\n'),
        # Fix extra closing braces
        (r'(\}\s*){4,}$', '}'),
    ]
    
    for pattern, replacement in replacements:
        content = re.sub(pattern, replacement, content, flags=re.MULTILINE | re.DOTALL)
    
    return content

def comprehensive_fix(file_path):
    """Apply all fixes to a file"""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        original = content
        
        # Apply fixes in specific order
        content = fix_interfaces_and_types(content)
        content = fix_object_properties_in_context(content)
        content = fix_specific_patterns(content)
        
        # Ensure file ends with newline
        if not content.endswith('\n'):
            content += '\n'
        
        if content != original:
            with open(file_path, 'w', encoding='utf-8') as f:
                f.write(content)
            return True
        return False
    except Exception as e:
        print(f"Error processing {file_path}: {e}")
        return False

def main():
    """Main function"""
    frontend_dir = Path(r"C:\Users\Hp\projects\ytempire-mvp\frontend")
    
    # Get all TypeScript files
    all_files = list(frontend_dir.glob("src/**/*.ts")) + list(frontend_dir.glob("src/**/*.tsx"))
    
    print(f"Processing {len(all_files)} TypeScript files...")
    
    fixed = 0
    for file_path in all_files:
        if comprehensive_fix(str(file_path)):
            fixed += 1
            if fixed % 10 == 0:
                print(f"  Fixed {fixed} files...")
    
    print(f"\n=== Complete ===")
    print(f"Total files processed: {len(all_files)}")
    print(f"Files fixed: {fixed}")

if __name__ == "__main__":
    main()