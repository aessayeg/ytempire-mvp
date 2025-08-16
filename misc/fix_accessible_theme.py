#!/usr/bin/env python3
"""
Fix syntax errors in accessibleTheme.ts
"""

import re
from pathlib import Path

def fix_accessible_theme(file_path):
    """Fix common syntax errors in accessibleTheme.ts"""
    
    with open(file_path, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # Fix hex colors with spaces
    content = re.sub(r'#([0-9A-Fa-f]) ([0-9A-Fa-f]{2}) ([0-9A-Fa-f]{2})\b', r'#\1\2\3', content)
    content = re.sub(r'#([0-9A-Fa-f]{2}) ([0-9A-Fa-f]{2}) ([0-9A-Fa-f])\b', r'#\1\2\3', content)
    content = re.sub(r'#([0-9A-Fa-f]{2})([0-9A-Fa-f]{2}) ([0-9A-Fa-f]{2})\b', r'#\1\2\3', content)
    content = re.sub(r'#([0-9A-Fa-f]) ([0-9A-Fa-f]{2}) ([0-9A-Fa-f]{2}) ([0-9A-Fa-f])\b', r'#\1\2\3\4', content)
    content = re.sub(r'#([0-9A-Fa-f]{2})([0-9A-Fa-f]{2}) ([A-Z0-9])', r'#\1\2\3', content)
    content = re.sub(r'#([0-9A-Fa-f]{3}) ([0-9A-Fa-f]{2})', r'#\1\2', content)
    
    # Fix letterSpacing with space before 'em'
    content = re.sub(r"letterSpacing: '(-?[0-9.]+) em'", r"letterSpacing: '\1em'", content)
    
    # Fix function syntax issues
    content = re.sub(r'checks\.forEach\(_\(\{ fg, bg, name \}\)', r'checks.forEach(({ fg, bg, name })', content)
    
    # Fix object property issues - commas in wrong places
    content = re.sub(r'\{ main:', r'{\n    main:', content)
    content = re.sub(r' \},\n  ', r'\n  },\n  ', content)
    
    # Fix missing semicolons
    content = re.sub(r'(console\.warn\([^)]+\))\}', r'\1;\n    }\n  });', content)
    
    # Fix the themeOptions declaration issue
    content = re.sub(r'const themeOptions: ThemeOptions = \{,', r'const themeOptions: ThemeOptions = {', content)
    
    # Fix object syntax where properties have wrong separators
    content = re.sub(r'const accessibleColors = \{ primary: \{', r'const accessibleColors = {\n  primary: {', content)
    
    # Fix typography section
    content = re.sub(r'typography: \{ // Ensure readable font sizes,\n  fontSize:', r'typography: {\n      // Ensure readable font sizes\n      fontSize:', content)
    
    # Fix components section
    content = re.sub(r'components: \{ // Ensure all interactive elements have proper focus indicators,\n  MuiButton:', 
                    r'components: {\n      // Ensure all interactive elements have proper focus indicators\n      MuiButton:', content)
    
    # Fix individual component style overrides
    content = re.sub(r'MuiButton: \{\n\n        styleOverrides: \{\n  root: \{', 
                    r'MuiButton: {\n        styleOverrides: {\n          root: {', content)
    content = re.sub(r'MuiIconButton: \{ styleOverrides: \{\n  root: \{', 
                    r'MuiIconButton: {\n        styleOverrides: {\n          root: {', content)
    content = re.sub(r'MuiTextField: \{ styleOverrides: \{\n  root: \{', 
                    r'MuiTextField: {\n        styleOverrides: {\n          root: {', content)
    content = re.sub(r'MuiLink: \{ styleOverrides: \{\n  root: \{', 
                    r'MuiLink: {\n        styleOverrides: {\n          root: {', content)
    content = re.sub(r'MuiChip: \{ styleOverrides: \{\n  root: \{', 
                    r'MuiChip: {\n        styleOverrides: {\n          root: {', content)
    content = re.sub(r'MuiCheckbox: \{ styleOverrides: \{\n  root: \{', 
                    r'MuiCheckbox: {\n        styleOverrides: {\n          root: {', content)
    content = re.sub(r'MuiRadio: \{ styleOverrides: \{\n  root: \{', 
                    r'MuiRadio: {\n        styleOverrides: {\n          root: {', content)
    content = re.sub(r'MuiSwitch: \{ styleOverrides: \{\n  root: \{', 
                    r'MuiSwitch: {\n        styleOverrides: {\n          root: {', content)
    content = re.sub(r'MuiListItem: \{ styleOverrides: \{\n  root: \{', 
                    r'MuiListItem: {\n        styleOverrides: {\n          root: {', content)
    content = re.sub(r'MuiMenuItem: \{ styleOverrides: \{\n  root: \{', 
                    r'MuiMenuItem: {\n        styleOverrides: {\n          root: {', content)
    content = re.sub(r'MuiOutlinedInput: \{ styleOverrides: \{\n  root: \{', 
                    r'MuiOutlinedInput: {\n        styleOverrides: {\n          root: {', content)
    
    # Fix closing braces issues for component styleOverrides
    content = re.sub(r'              outlineOffset: 2 \}\n          \}\n        \}\n      \}', 
                    r'              outlineOffset: 2\n            }\n          }\n        }\n      },', content)
    
    # Fix shape object
    content = re.sub(r'    shape: \{ borderRadius: 4 \},', r'    shape: {\n      borderRadius: 4\n    },', content)
    
    # Fix the ending of the file
    content = re.sub(r'export const createOptimizedRouter.*?return createBrowserRouter.*?\]\)\}', 
                    r'export const lightAccessibleTheme = createAccessibleTheme(\'light\');\nexport const darkAccessibleTheme = createAccessibleTheme(\'dark\');', 
                    content, flags=re.DOTALL)
    
    # Write back
    with open(file_path, 'w', encoding='utf-8') as f:
        f.write(content)
    
    print(f"Fixed {file_path}")

if __name__ == "__main__":
    theme_path = Path(r"C:\Users\Hp\projects\ytempire-mvp\frontend\src\theme\accessibleTheme.ts")
    if theme_path.exists():
        fix_accessible_theme(theme_path)