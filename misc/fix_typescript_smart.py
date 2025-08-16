#!/usr/bin/env python3
"""
Smart TypeScript error fixer that preserves style objects
"""

import re
import os
from pathlib import Path
import glob

def fix_typescript_smart(content):
    """Fix TypeScript errors more carefully"""
    
    # Fix 1: Fix semicolons in object properties that were incorrectly changed
    # Style objects should use commas, not semicolons
    # Look for patterns inside sx, style, or CSS-in-JS objects
    
    # First, identify style object contexts
    style_blocks = []
    
    # Find sx={{ ... }} blocks
    sx_pattern = r'sx=\{\{([^}]*)\}\}'
    for match in re.finditer(sx_pattern, content, re.DOTALL):
        style_blocks.append((match.start(1), match.end(1)))
    
    # Find style={{ ... }} blocks
    style_pattern = r'style=\{\{([^}]*)\}\}'
    for match in re.finditer(style_pattern, content, re.DOTALL):
        style_blocks.append((match.start(1), match.end(1)))
    
    # Convert content to list for easier manipulation
    content_list = list(content)
    
    # In style blocks, replace semicolons back to commas
    for start, end in style_blocks:
        block = content[start:end]
        # Replace semicolons with commas in property definitions
        fixed_block = re.sub(r';(\s*\n\s*[\'"\w])', r',\1', block)
        # Remove trailing semicolon before closing brace
        fixed_block = re.sub(r';(\s*$)', r'\1', fixed_block)
        
        # Replace in content
        for i, char in enumerate(fixed_block):
            content_list[start + i] = char
    
    content = ''.join(content_list)
    
    # Fix 2: Fix object literal syntax issues
    # Fix patterns like {, to {
    content = re.sub(r'\{,\s*\n', r'{\n', content)
    
    # Fix 3: Fix double backticks in template literals
    content = re.sub(r'``', r'`', content)
    
    # Fix 4: Fix semicolons that should be commas in object literals
    # But NOT in interfaces or types
    # Look for object literal contexts
    content = re.sub(r'(\w+):\s*([^;,\n{}]+);(\s*\n\s*\w+:)', r'\1: \2,\3', content)
    
    # Fix 5: Fix missing closing braces in functions
    # Look for patterns like );` at end of line
    content = re.sub(r'\);`\s*$', r');\n};', content, flags=re.MULTILINE)
    
    # Fix 6: Fix extra closing braces
    content = re.sub(r'\}\}\}\}\}\}\}', r'}', content)
    
    # Fix 7: Fix unterminated lines in objects
    content = re.sub(r'(\w+):\s*([^;,\n{}]+)``', r'\1: \2`', content)
    
    # Fix 8: Fix broken array syntax  
    content = re.sub(r'(\[\s*);', r'\1', content)
    content = re.sub(r';(\s*\])', r'\1', content)
    
    # Fix 9: Remove "No newline at end of file" text
    content = re.sub(r'\s*No newline at end of file', '', content)
    
    return content

def fix_file_smart(file_path):
    """Fix a single TypeScript/TSX file smartly"""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        fixed_content = fix_typescript_smart(content)
        
        if fixed_content != content:
            with open(file_path, 'w', encoding='utf-8') as f:
                f.write(fixed_content)
            return True
        return False
    except Exception as e:
        print(f"Error processing {file_path}: {e}")
        return False

def main():
    """Main function to fix TypeScript files smartly"""
    
    frontend_dir = Path(r"C:\Users\Hp\projects\ytempire-mvp\frontend")
    
    # Files that were recently modified and need fixing
    problem_files = [
        "src/components/Animations/styledComponents.ts",
        "src/components/Animations/variants.ts",
        "src/components/Animations/AdvancedAnimations.tsx",
        "src/components/Analytics/CompetitiveAnalysisDashboard.tsx",
        "src/components/Analytics/UserBehaviorDashboard.tsx",
        "src/stores/optimizedStore.ts",
        "src/stores/videoStore.ts",
        "src/stores/authStore.ts",
        "src/theme/accessibleTheme.ts",
    ]
    
    total_fixed = 0
    
    for file_rel in problem_files:
        file_path = frontend_dir / file_rel
        if file_path.exists():
            if fix_file_smart(str(file_path)):
                total_fixed += 1
                print(f"Fixed: {file_rel}")
    
    print(f"\n=== Summary ===")
    print(f"Files fixed: {total_fixed}")

if __name__ == "__main__":
    main()