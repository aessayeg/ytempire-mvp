#!/usr/bin/env python3
"""
Comprehensive TypeScript error fixer - meticulous approach
"""

import re
import os
from pathlib import Path

def fix_interfaces_properly(content):
    """Fix interface definitions properly"""
    # Fix interface property separators - use semicolons
    def fix_interface_props(match):
        interface_name = match.group(1)
        interface_body = match.group(2)
        
        # Split by lines and fix each property
        lines = interface_body.split('\n')
        fixed_lines = []
        
        for line in lines:
            line = line.strip()
            if line and ':' in line:
                # Remove trailing comma or semicolon, will add semicolon
                line = re.sub(r'[,;]\s*$', '', line)
                # Don't add semicolon to the last property if it's before }
                if not line.endswith('}'):
                    line += ';'
            fixed_lines.append(line)
        
        return f'interface {interface_name} {{\n  {chr(10).join(fixed_lines)}\n}}'
    
    content = re.sub(
        r'interface\s+(\w+)\s*\{([^}]*)\}',
        fix_interface_props,
        content,
        flags=re.DOTALL
    )
    
    return content

def fix_style_props(content):
    """Fix sx and style props to use commas correctly"""
    def fix_sx_style(match):
        prop_name = match.group(1)
        prop_content = match.group(2)
        
        # Inside sx/style props, use commas not semicolons
        fixed = prop_content
        # Replace semicolons with commas between properties
        fixed = re.sub(r';(\s*[\w"\'])', r',\1', fixed)
        # Remove trailing semicolon/comma
        fixed = re.sub(r'[;,]\s*$', '', fixed)
        
        return f'{prop_name}={{{{{fixed}}}}}'
    
    # Fix sx={{ ... }} and style={{ ... }}
    content = re.sub(
        r'((?:sx|style)\s*)=\{\{([^}]*?)\}\}',
        fix_sx_style,
        content,
        flags=re.DOTALL
    )
    
    return content

def fix_object_literals(content):
    """Fix object literal syntax"""
    # Fix array of objects
    def fix_array_objects(match):
        array_content = match.group(1)
        # In arrays, objects should use commas
        fixed = re.sub(r';(\s*[\w"\'])', r',\1', array_content)
        # Remove trailing semicolon before }
        fixed = re.sub(r';\s*\}', '}', fixed)
        return '[' + fixed + ']'
    
    content = re.sub(
        r'\[([^\[\]]*)\]',
        fix_array_objects,
        content
    )
    
    # Fix object assignments
    def fix_object_assign(match):
        var_part = match.group(1)
        obj_content = match.group(2)
        # In object literals, use commas
        fixed = re.sub(r';(\s*[\w"\'])', r',\1', obj_content)
        fixed = re.sub(r';\s*\}', '}', fixed)
        return var_part + ' = {' + fixed + '}'
    
    content = re.sub(
        r'(const\s+\w+(?::\s*[^=]+)?|let\s+\w+(?::\s*[^=]+)?|var\s+\w+(?::\s*[^=]+)?)\s*=\s*\{([^{}]*)\}',
        fix_object_assign,
        content
    )
    
    return content

def fix_jsx_syntax(content):
    """Fix JSX syntax issues"""
    # Fix incomplete closing tags
    content = re.sub(r'}\s*\)\s*}\s*$', '})}', content, flags=re.MULTILINE)
    
    # Fix misplaced braces in JSX
    content = re.sub(r'\}\s*\n\s*\}\)\}', '})}', content)
    
    # Fix extra closing braces
    content = re.sub(r'\}\s*;\s*$', '};', content, flags=re.MULTILINE)
    
    return content

def fix_template_literals(content):
    """Fix template literal issues"""
    # Remove double backticks
    content = re.sub(r'``+', '`', content)
    
    # Fix backticks at end of lines
    content = re.sub(r'`\s*$', '', content, flags=re.MULTILINE)
    
    # Fix keyframes syntax
    content = re.sub(r'(keyframes)\s*`([^`]+)`\s*;', r'\1`\2`;', content)
    
    return content

def fix_function_syntax(content):
    """Fix function and arrow function syntax"""
    # Fix arrow functions with broken syntax
    content = re.sub(r'\)\s*=>\s*\}', r') => {}', content)
    
    # Fix missing closing braces
    content = re.sub(r'(\w+)\s*:\s*\(\)\s*=>\s*\{([^}]*)\n\s*$', r'\1: () => {\2\n}', content, flags=re.MULTILINE)
    
    return content

def remove_junk(content):
    """Remove junk text and fix formatting"""
    # Remove "No newline at end of file" text
    content = re.sub(r'\s*No newline at end of file\s*', '\n', content)
    
    # Fix multiple closing braces at end
    content = re.sub(r'(\}\s*){4,}$', '}', content)
    
    # Fix duplicate exports
    content = re.sub(r'(export\s+default\s+\{[^}]+\})\s*;\s*\1', r'\1', content)
    
    # Ensure newline at end
    if not content.endswith('\n'):
        content += '\n'
    
    return content

def fix_typescript_file(file_path):
    """Fix a single TypeScript file comprehensively"""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        original = content
        
        # Apply all fixes in order
        content = fix_interfaces_properly(content)
        content = fix_style_props(content)
        content = fix_object_literals(content)
        content = fix_jsx_syntax(content)
        content = fix_template_literals(content)
        content = fix_function_syntax(content)
        content = remove_junk(content)
        
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
    
    # Priority files with most errors
    priority_files = [
        "src/components/Analytics/CompetitiveAnalysisDashboard.tsx",
        "src/components/Analytics/AnalyticsDashboard.tsx",
        "src/components/Analytics/UserBehaviorDashboard.tsx",
        "src/components/Animations/AdvancedAnimations.tsx",
        "src/components/Accessibility/AccessibleButton.tsx",
        "src/components/Accessibility/ScreenReaderAnnouncer.tsx",
        "src/components/Accessibility/SkipNavigation.tsx",
    ]
    
    print("Fixing priority files...")
    fixed_count = 0
    
    for file_rel in priority_files:
        file_path = frontend_dir / file_rel
        if file_path.exists():
            if fix_typescript_file(str(file_path)):
                fixed_count += 1
                print(f"  Fixed: {file_rel}")
    
    # Now process all other TypeScript files
    print("\nProcessing all TypeScript files...")
    
    all_ts_files = list(frontend_dir.glob("src/**/*.ts"))
    all_tsx_files = list(frontend_dir.glob("src/**/*.tsx"))
    all_files = all_ts_files + all_tsx_files
    
    for file_path in all_files:
        if fix_typescript_file(str(file_path)):
            fixed_count += 1
            print(f"  Fixed: {file_path.relative_to(frontend_dir)}")
    
    print(f"\n=== Summary ===")
    print(f"Total files fixed: {fixed_count}")

if __name__ == "__main__":
    main()