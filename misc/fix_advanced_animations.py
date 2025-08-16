#!/usr/bin/env python3
"""
Fix syntax errors in AdvancedAnimations.tsx
"""

import re
from pathlib import Path

def fix_advanced_animations(file_path):
    """Fix common syntax errors in AdvancedAnimations.tsx"""
    
    with open(file_path, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # Fix React.FC<{, pattern to React.FC<{
    content = re.sub(r'React\.FC<\{,', r'React.FC<{', content)
    
    # Fix missing closing brace in interface definitions
    content = re.sub(r'className\?\: string\}>', r'className?: string;\n}>', content)
    content = re.sub(r'onClick\?\: \(\) => void\}>', r'onClick?: () => void;\n}>', content)
    
    # Fix missing semicolons in interfaces
    content = re.sub(r'(children: React\.ReactNode)\n', r'\1;\n', content)
    content = re.sub(r'(delay\?: number)\n', r'\1;\n', content)
    content = re.sub(r'(duration\?: number)\n', r'\1;\n', content)
    content = re.sub(r'(text: string)\n', r'\1;\n', content)
    content = re.sub(r'(from: number)\n', r'\1;\n', content)
    content = re.sub(r'(to: number)\n', r'\1;\n', content)
    content = re.sub(r'(speed\?: number)\n', r'\1;\n', content)
    content = re.sub(r'(label: string)\n', r'\1;\n', content)
    content = re.sub(r'(color\?: string)\n', r'\1;\n', content)
    content = re.sub(r'(size\?: string)\n', r'\1;\n', content)
    content = re.sub(r'(className\?: string);', r'className?: string;', content)
    
    # Write back
    with open(file_path, 'w', encoding='utf-8') as f:
        f.write(content)
    
    print(f"Fixed {file_path}")

if __name__ == "__main__":
    animations_path = Path(r"C:\Users\Hp\projects\ytempire-mvp\frontend\src\components\Animations\AdvancedAnimations.tsx")
    if animations_path.exists():
        fix_advanced_animations(animations_path)