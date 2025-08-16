#!/usr/bin/env python3
"""
Fix remaining TypeScript errors in the frontend
"""

import re
import os
from pathlib import Path
import glob

def fix_typescript_comprehensive(content):
    """Fix remaining TypeScript errors comprehensively"""
    
    # Fix 1: Fix object property syntax in style props
    # Match patterns like property: value; in sx and style props
    def fix_style_syntax(match):
        prop_content = match.group(2)
        # Replace semicolons with commas, but keep the last one without comma
        fixed = re.sub(r';(\s*["\'\w])', r',\1', prop_content)
        # Remove trailing semicolon/comma
        fixed = re.sub(r'[;,]\s*$', '', fixed)
        return match.group(1) + '={{' + fixed + '}}'
    
    content = re.sub(r'((?:sx|style)\s*)=\{\{([^}]*)\}\}', fix_style_syntax, content, flags=re.DOTALL)
    
    # Fix 2: Fix template literal backticks
    content = re.sub(r'`{2,}', '`', content)
    content = re.sub(r'`\s*\n\s*}', '}', content)
    content = re.sub(r'`\s*$', '', content, flags=re.MULTILINE)
    
    # Fix 3: Fix object/array syntax
    # Fix {, patterns
    content = re.sub(r'\{,\s*', '{\n  ', content)
    # Fix [, patterns
    content = re.sub(r'\[,\s*', '[\n  ', content)
    # Fix trailing semicolons in objects
    content = re.sub(r';(\s*\})', r'\1', content)
    # Fix trailing semicolons in arrays
    content = re.sub(r';(\s*\])', r'\1', content)
    
    # Fix 4: Fix styled components template literals
    content = re.sub(r'(styled\([^)]+\))\s*([^`])', r'\1`\2', content)
    content = re.sub(r'(keyframes)\s*([^`])', r'\1`\2', content)
    
    # Fix 5: Fix color codes with spaces
    content = re.sub(r'#([0-9a-fA-F]{3,6})\s+([0-9a-fA-F]+)', r'#\1\2', content)
    content = re.sub(r'#([0-9a-fA-F]{1,2})\s+([0-9a-fA-F]{1,2})\s+([0-9a-fA-F]{1,2})', r'#\1\2\3', content)
    
    # Fix 6: Fix interface/type definitions
    # Ensure interfaces use semicolons
    def fix_interface(match):
        interface_content = match.group(2)
        # Replace commas with semicolons between properties
        fixed = re.sub(r',(\s*\n\s*\w+\s*:)', r';\1', interface_content)
        # Remove trailing comma/semicolon
        fixed = re.sub(r'[,;]\s*$', '', fixed)
        return 'interface ' + match.group(1) + ' {' + fixed + '\n}'
    
    content = re.sub(r'interface\s+(\w+)\s*\{([^}]*)\}', fix_interface, content, flags=re.DOTALL)
    
    # Fix 7: Fix type definitions
    def fix_type(match):
        type_content = match.group(2)
        # Replace commas with semicolons between properties
        fixed = re.sub(r',(\s*\n\s*\w+\s*:)', r';\1', type_content)
        # Remove trailing comma/semicolon
        fixed = re.sub(r'[,;]\s*$', '', fixed)
        return 'type ' + match.group(1) + ' = {' + fixed + '\n}'
    
    content = re.sub(r'type\s+(\w+)\s*=\s*\{([^}]*)\}', fix_type, content, flags=re.DOTALL)
    
    # Fix 8: Fix arrow function returns
    # Fix patterns like ) => }
    content = re.sub(r'\)\s*=>\s*\}', r') => {}', content)
    
    # Fix 9: Fix JSX closing tags
    # Ensure closing tags are complete
    content = re.sub(r'<\/([A-Z]\w*)\s*$', r'</\1>', content, flags=re.MULTILINE)
    
    # Fix 10: Fix unterminated template literals
    # Add closing backtick if missing
    lines = content.split('\n')
    in_template = False
    fixed_lines = []
    
    for line in lines:
        backtick_count = line.count('`')
        if backtick_count % 2 == 1:
            in_template = not in_template
        
        # If we're ending a line in a template without closing it
        if in_template and '`' not in line and not line.strip().endswith('\\'):
            # Check if next line starts a new statement
            if re.match(r'^\s*(const|let|var|function|class|export|import|if|for|while|return)', line):
                # Close the template on previous line
                if fixed_lines:
                    fixed_lines[-1] += '`'
                in_template = False
        
        fixed_lines.append(line)
    
    content = '\n'.join(fixed_lines)
    
    # Fix 11: Remove "No newline at end of file" text
    content = re.sub(r'\s*No newline at end of file\s*', '\n', content)
    
    # Fix 12: Fix duplicate closing braces at end of file
    content = re.sub(r'(\}\s*){4,}$', '}', content)
    
    # Fix 13: Ensure file ends with newline
    if not content.endswith('\n'):
        content += '\n'
    
    return content

def process_file(file_path):
    """Process a single TypeScript/TSX file"""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        fixed_content = fix_typescript_comprehensive(content)
        
        if fixed_content != content:
            with open(file_path, 'w', encoding='utf-8') as f:
                f.write(fixed_content)
            return True
        return False
    except Exception as e:
        print(f"Error processing {file_path}: {e}")
        return False

def main():
    """Main function to fix TypeScript files"""
    
    frontend_dir = Path(r"C:\Users\Hp\projects\ytempire-mvp\frontend")
    
    # Process all TypeScript and TSX files
    ts_files = list(frontend_dir.glob("src/**/*.ts"))
    tsx_files = list(frontend_dir.glob("src/**/*.tsx"))
    
    all_files = ts_files + tsx_files
    
    print(f"Found {len(all_files)} TypeScript files to process")
    
    total_fixed = 0
    batch_size = 50
    
    for i in range(0, len(all_files), batch_size):
        batch = all_files[i:i+batch_size]
        print(f"\nProcessing batch {i//batch_size + 1} ({i+1}-{min(i+batch_size, len(all_files))} of {len(all_files)})")
        
        for file_path in batch:
            if process_file(str(file_path)):
                total_fixed += 1
                print(f"  Fixed: {file_path.relative_to(frontend_dir)}")
    
    print(f"\n=== Summary ===")
    print(f"Total files processed: {len(all_files)}")
    print(f"Files fixed: {total_fixed}")

if __name__ == "__main__":
    main()