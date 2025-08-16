#!/usr/bin/env python3
"""
Targeted TypeScript error fixer for specific patterns
"""

import re
import os
from pathlib import Path

def fix_typescript_targeted(content):
    """Fix specific TypeScript error patterns"""
    
    # Fix 1: Fix object literal issues in interfaces/types
    # Look for patterns like "property: value;" in interfaces
    def fix_interface_syntax(match):
        inside = match.group(1)
        # In interfaces, use semicolons
        fixed = re.sub(r',(\s*\n\s*\w+:)', r';\1', inside)
        return 'interface ' + match.group(2) + ' {' + fixed + '}'
    
    content = re.sub(r'interface\s+(\w+)\s*\{([^}]*)\}', fix_interface_syntax, content, flags=re.DOTALL)
    
    # Fix 2: Fix object literals in arrays (should use commas)
    # Pattern: {property: value; property: value}
    def fix_object_in_array(match):
        obj_content = match.group(1)
        # Replace semicolons with commas in object literals
        fixed = re.sub(r';(\s*(?:\w+|["\'])[^:]*:)', r',\1', obj_content)
        # Remove trailing semicolon before }
        fixed = re.sub(r';\s*$', '', fixed)
        return '{' + fixed + '}'
    
    # Find patterns like [{...}] or = {...}
    content = re.sub(r'(?<=[=\[\(,])\s*\{([^{}]*)\}', fix_object_in_array, content)
    
    # Fix 3: Fix style/sx props - these should always use commas
    # Match sx={{ ... }} or style={{ ... }}
    def fix_style_props(match):
        prop_name = match.group(1)
        inner_content = match.group(2)
        # Replace all semicolons with commas
        fixed = inner_content.replace(';', ',')
        # Remove trailing comma before closing brace
        fixed = re.sub(r',\s*$', '', fixed)
        return f'{prop_name}={{{{{fixed}}}}}'
    
    content = re.sub(r'(sx|style)=\{\{([^}]*)\}\}', fix_style_props, content, flags=re.DOTALL)
    
    # Fix 4: Fix spaces in URLs (common in placeholder URLs)
    content = re.sub(r'https:\s+//', 'https://', content)
    content = re.sub(r'http:\s+//', 'http://', content)
    
    # Fix 5: Fix duplicate closing braces and backticks
    content = re.sub(r'`+\s*\}+\s*;?\s*$', '', content, flags=re.MULTILINE)
    content = re.sub(r'\}\s*\}\s*\}\s*\}\s*\}\s*\}+', '}', content)
    
    # Fix 6: Fix color codes with spaces
    content = re.sub(r'#([0-9a-fA-F]{3})\s+([0-9a-fA-F]{1,3})\b', r'#\1\2', content)
    
    # Fix 7: Fix trailing backticks at end of lines
    content = re.sub(r'`\s*$', '', content, flags=re.MULTILINE)
    
    # Fix 8: Fix duplicate semicolons
    content = re.sub(r';;+', ';', content)
    
    # Fix 9: Fix broken template literals
    content = re.sub(r'``+', '`', content)
    
    # Fix 10: Fix "No newline at end of file" text
    content = re.sub(r'\s*No newline at end of file\s*$', '\n', content)
    
    # Fix 11: Remove extra closing braces at end of file
    if content.count('{') < content.count('}'):
        # Count the difference and remove extra closing braces from the end
        diff = content.count('}') - content.count('{')
        if diff > 0:
            # Remove extra closing braces from the end
            content = re.sub(r'\}+\s*$', lambda m: '}' * max(0, len(m.group()) - diff), content)
    
    return content

def process_file(file_path):
    """Process a single TypeScript/TSX file"""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        fixed_content = fix_typescript_targeted(content)
        
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
    
    # Files with the most errors based on build output
    priority_files = [
        "src/components/Analytics/CompetitiveAnalysisDashboard.tsx",
        "src/components/Analytics/AnalyticsDashboard.tsx",
        "src/components/Analytics/UserBehaviorDashboard.tsx",
        "src/components/Accessibility/AccessibleButton.tsx",
        "src/components/Accessibility/ScreenReaderAnnouncer.tsx",
        "src/components/Accessibility/SkipNavigation.tsx",
        "src/components/Animations/styledComponents.ts",
        "src/components/Animations/variants.ts",
        "src/components/Animations/AdvancedAnimations.tsx",
    ]
    
    total_fixed = 0
    
    print("Processing priority files...")
    for file_rel in priority_files:
        file_path = frontend_dir / file_rel
        if file_path.exists():
            if process_file(str(file_path)):
                total_fixed += 1
                print(f"Fixed: {file_rel}")
    
    print(f"\n=== Summary ===")
    print(f"Files fixed: {total_fixed}")

if __name__ == "__main__":
    main()