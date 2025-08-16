#!/usr/bin/env python3
"""
Comprehensive TypeScript error fixer for common syntax issues
Fixes TS1005 (missing punctuation) and TS1128 (declaration expected) errors
"""

import re
import os
from pathlib import Path
import glob

def fix_typescript_syntax(content):
    """Fix common TypeScript syntax errors"""
    
    # Store original for comparison
    original = content
    
    # Fix 1: React.FC<{, pattern (comma after opening brace)
    content = re.sub(r'React\.FC<\{,', r'React.FC<{', content)
    
    # Fix 2: Interface/type properties with commas instead of semicolons
    # Fix patterns like "property: type," to "property: type;"
    content = re.sub(r'(\w+\s*:\s*[^,;{}]+),(\s*\n\s*\w+\s*:)', r'\1;\2', content)
    
    # Fix 3: Object literals with {, pattern
    content = re.sub(r'\{,(\s*\n)', r'{\1', content)
    content = re.sub(r'\{ ,(\s*\n)', r'{\1', content)
    
    # Fix 4: Hex colors with spaces
    content = re.sub(r'#([0-9A-Fa-f]{1,2})\s+([0-9A-Fa-f]{2})\s+([0-9A-Fa-f]{1,2})', r'#\1\2\3', content)
    content = re.sub(r'#([0-9A-Fa-f]{3})\s+([0-9A-Fa-f]{3})', r'#\1\2', content)
    content = re.sub(r'#([0-9A-Fa-f]{2})([0-9A-Fa-f]{2})\s+([A-Z0-9])', r'#\1\2\3', content)
    
    # Fix 5: Arrow functions with wrong syntax
    content = re.sub(r'=>\s*\{\}(\s*\n\s+)', r'=>\1', content)
    content = re.sub(r'useCallback\(_\(', r'useCallback((', content)
    content = re.sub(r'forEach\(_\(\{', r'forEach(({', content)
    content = re.sub(r'map\(_\(\{', r'map(({', content)
    content = re.sub(r'filter\(_\(', r'filter((', content)
    
    # Fix 6: Missing semicolons at end of statements
    # Add semicolon after closing parenthesis if followed by newline
    content = re.sub(r'(\))(\s*\n\s*(?:const|let|var|function|export|import|if|for|while|return))', r'\1;\2', content)
    
    # Fix 7: Template literals issues
    # Fix unterminated template literals (missing closing backtick)
    content = re.sub(r'`([^`]*)\n\s*}', r'`\1`\n  }', content)
    
    # Fix 8: useState and other hooks with wrong syntax
    content = re.sub(r'useState<([^>]+)>\(\{,', r'useState<\1>({', content)
    content = re.sub(r'useRef<([^>]+)>\(\{,', r'useRef<\1>({', content)
    content = re.sub(r'useMemo<([^>]+)>\(\{,', r'useMemo<\1>({', content)
    
    # Fix 9: Interface definitions with wrong separators
    # Fix interface properties ending with comma
    content = re.sub(r'(\s+)(\w+\??\s*:\s*[^;,\n{}]+),(\s*\n\s*})', r'\1\2;\3', content)
    
    # Fix 10: Function parameters with wrong syntax
    content = re.sub(r'\(\s*\{,', r'({', content)
    content = re.sub(r',\s*\}\s*:', r'}: ', content)
    
    # Fix 11: JSX props with spaces in values
    content = re.sub(r'value="(\d+)\s+(\w)"', r'value="\1\2"', content)
    
    # Fix 12: Missing closing braces
    # Count opening and closing braces
    open_braces = content.count('{')
    close_braces = content.count('}')
    if open_braces > close_braces:
        # Add missing closing braces at the end
        content += '\n' + '}' * (open_braces - close_braces)
    
    # Fix 13: Variants and object definitions
    content = re.sub(r': Variants = \{,', r': Variants = {', content)
    content = re.sub(r'animate: \{,', r'animate: {', content)
    content = re.sub(r'initial: \{,', r'initial: {', content)
    content = re.sub(r'transition: \{,', r'transition: {', content)
    
    # Fix 14: Style objects with wrong syntax
    content = re.sub(r'sx=\{\{,', r'sx={{', content)
    content = re.sub(r'style=\{\{,', r'style={{', content)
    
    # Fix 15: letterSpacing and other CSS properties
    content = re.sub(r"'(-?[0-9.]+)\s+em'", r"'\1em'", content)
    content = re.sub(r"'(-?[0-9.]+)\s+rem'", r"'\1rem'", content)
    content = re.sub(r"'(-?[0-9.]+)\s+px'", r"'\1px'", content)
    
    # Fix 16: Function return statements
    content = re.sub(r'\)\s*\};', r');\n};', content)
    
    # Fix 17: Export statements
    content = re.sub(r'export default \{,', r'export default {', content)
    
    # Fix 18: Array/object destructuring
    content = re.sub(r'const \[\s*,', r'const [', content)
    content = re.sub(r'const \{\s*,', r'const {', content)
    content = re.sub(r'let \[\s*,', r'let [', content)
    content = re.sub(r'let \{\s*,', r'let {', content)
    
    # Fix 19: Switch/case statements
    content = re.sub(r'case\s+\'([^\']+)\'\s+:', r"case '\1':", content)
    content = re.sub(r'case\s+"([^"]+)"\s+:', r'case "\1":', content)
    
    # Fix 20: Async/await syntax
    content = re.sub(r'async\s+\(\s*\)', r'async ()', content)
    content = re.sub(r'await\s+\(\s*', r'await (', content)
    
    return content

def fix_file(file_path):
    """Fix a single TypeScript/TSX file"""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        fixed_content = fix_typescript_syntax(content)
        
        if fixed_content != content:
            with open(file_path, 'w', encoding='utf-8') as f:
                f.write(fixed_content)
            return True
        return False
    except Exception as e:
        print(f"Error processing {file_path}: {e}")
        return False

def main():
    """Main function to fix all TypeScript files"""
    
    frontend_dir = Path(r"C:\Users\Hp\projects\ytempire-mvp\frontend")
    
    # File patterns to fix
    patterns = [
        "src/**/*.ts",
        "src/**/*.tsx",
    ]
    
    total_fixed = 0
    total_files = 0
    
    for pattern in patterns:
        files = glob.glob(str(frontend_dir / pattern), recursive=True)
        for file_path in files:
            # Skip node_modules and build directories
            if 'node_modules' in file_path or 'dist' in file_path or 'build' in file_path:
                continue
            
            total_files += 1
            if fix_file(file_path):
                total_fixed += 1
                print(f"Fixed: {Path(file_path).relative_to(frontend_dir)}")
    
    print(f"\n=== Summary ===")
    print(f"Total files processed: {total_files}")
    print(f"Files fixed: {total_fixed}")
    print(f"Files unchanged: {total_files - total_fixed}")

if __name__ == "__main__":
    main()