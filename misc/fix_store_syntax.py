#!/usr/bin/env python3
"""
Fix syntax errors in store files
"""

import re
from pathlib import Path

def fix_store_syntax(file_path):
    """Fix common syntax errors in store files"""
    
    with open(file_path, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # Fix all instances of set(_(state) to set((state)
    content = re.sub(r'set\(_\(state\)', 'set((state)', content)
    
    # Fix all instances of => {} followed by code to =>
    content = re.sub(r'=>\s*\{\}\s*\n\s+set\(', r'=>\n            set(', content)
    
    # Fix closing braces without semicolons
    content = re.sub(r'(\s+)(state\.[a-zA-Z\.]+\s*=\s*[^;]+)\}\)', r'\1\2;\n            })', content)
    content = re.sub(r'(\s+)(Object\.assign\([^;]+)\}\)', r'\1\2;\n              }', content)
    content = re.sub(r'(\s+)([a-zA-Z\.]+\.push\([^;]+)\}\)', r'\1\2;\n            })', content)
    
    # Fix template literal issues
    content = re.sub(r'`Bearer \$\{localStorage\.getItem\(\'token\'\)\}\n', r"`Bearer ${localStorage.getItem('token')}`", content)
    
    # Write back
    with open(file_path, 'w', encoding='utf-8') as f:
        f.write(content)
    
    print(f"Fixed {file_path}")

if __name__ == "__main__":
    # Fix optimizedStore.ts
    store_path = Path(r"C:\Users\Hp\projects\ytempire-mvp\frontend\src\stores\optimizedStore.ts")
    if store_path.exists():
        fix_store_syntax(store_path)
    
    # Fix videoStore.ts
    video_store_path = Path(r"C:\Users\Hp\projects\ytempire-mvp\frontend\src\stores\videoStore.ts")
    if video_store_path.exists():
        fix_store_syntax(video_store_path)