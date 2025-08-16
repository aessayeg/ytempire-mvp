#!/usr/bin/env python3
"""
Fix syntax errors in variants.ts
"""

import re
from pathlib import Path

def fix_variants(file_path):
    """Fix common syntax errors in variants.ts"""
    
    with open(file_path, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # Fix Variants = {, pattern
    content = re.sub(r': Variants = \{,', r': Variants = {', content)
    
    # Fix object property issues - commas after opening braces
    content = re.sub(r'\{,\n', r'{\n', content)
    content = re.sub(r'\{ ,\n', r'{\n', content)
    
    # Fix transition: {, pattern
    content = re.sub(r'transition: \{,', r'transition: {', content)
    
    # Fix animate: {, pattern
    content = re.sub(r'animate: \{,', r'animate: {', content)
    content = re.sub(r'animate: \{ ,', r'animate: {', content)
    
    # Fix hover: {, pattern
    content = re.sub(r'hover: \{ ,', r'hover: {', content)
    
    # Fix initial: {, pattern
    content = re.sub(r'initial: \{,', r'initial: {', content)
    
    # Fix spaces in property values
    content = re.sub(r'opacity:\s+0', r'opacity: 0', content)
    content = re.sub(r'opacity:\s+1', r'opacity: 1', content)
    content = re.sub(r'scale:\s+', r'scale: ', content)
    content = re.sub(r'y:\s+', r'y: ', content)
    content = re.sub(r'x:\s+', r'x: ', content)
    
    # Fix closing brace issue
    content = re.sub(r'\n  \}\n\};', r'\n  }\n};', content)
    
    # Write back
    with open(file_path, 'w', encoding='utf-8') as f:
        f.write(content)
    
    print(f"Fixed {file_path}")

if __name__ == "__main__":
    variants_path = Path(r"C:\Users\Hp\projects\ytempire-mvp\frontend\src\components\Animations\variants.ts")
    if variants_path.exists():
        fix_variants(variants_path)