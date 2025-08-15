#!/usr/bin/env python3
"""
Fix specific syntax errors in frontend components
"""

import os
import re
from pathlib import Path

def fix_file_specific_errors():
    """Fix specific errors in known problematic files"""
    
    fixes = {
        # AccessibleButton.tsx - Unterminated template literal at line 86
        "frontend/src/components/Accessibility/AccessibleButton.tsx": [
            (r"(\$\{[^}]*)`([^`]*$)", r"\1}\2", "Fix unterminated template literal"),
            (r"(`[^`]*\$\{[^}]*)$", r"\1}`", "Close template literal"),
        ],
        
        # AnalyticsDashboard.tsx - JSX must have one parent at line 650
        "frontend/src/components/Analytics/AnalyticsDashboard.tsx": [
            (r"return\s+\(\s*\n\s*<(\w+)", r"return (\n    <React.Fragment>\n      <\1", "Wrap in Fragment"),
            (r"(<\/\w+>\s*\n\s*<\w+)", r"\1", "Fix sibling elements"),
        ],
        
        # CompetitiveAnalysisDashboard.tsx - semicolon expected at 169
        "frontend/src/components/Analytics/CompetitiveAnalysisDashboard.tsx": [
            (r"}\s*\n\s*interface", r"};\n\ninterface", "Add semicolon before interface"),
            (r"}\s*\n\s*const", r"};\n\nconst", "Add semicolon before const"),
        ],
        
        # UserBehaviorDashboard.tsx - comma expected at 151
        "frontend/src/components/Analytics/UserBehaviorDashboard.tsx": [
            (r"(\w+:\s*[^,}\n]+)\n\s*(\w+:)", r"\1,\n  \2", "Add missing comma"),
        ],
        
        # variants.ts - comma expected
        "frontend/src/components/Animations/variants.ts": [
            (r"(\w+:\s*{[^}]*})\n\s*(\w+:)", r"\1,\n  \2", "Add comma between variants"),
        ],
        
        # EmailVerification.tsx - comma expected at 22
        "frontend/src/components/Auth/EmailVerification.tsx": [
            (r"(useState\([^)]*\))\s*\n", r"\1;\n", "Add semicolon after useState"),
        ],
        
        # ForgotPasswordForm.tsx - semicolon expected at 105
        "frontend/src/components/Auth/ForgotPasswordForm.tsx": [
            (r"(setSubmitting\(false\))\s*$", r"\1;", "Add semicolon"),
        ],
        
        # TwoFactorAuth.tsx - closing brace expected at 176
        "frontend/src/components/Auth/TwoFactorAuth.tsx": [
            (r"(\{[^}]*)\s*$", r"\1\n}", "Add closing brace"),
        ],
        
        # BatchOperations.tsx - statement expected
        "frontend/src/components/BatchOperations/BatchOperations.tsx": [
            (r"^(\s*)}\s*else\s*{", r"\1} else {", "Fix else statement"),
            (r";\s*}\s*;", r";}", "Remove extra semicolon"),
        ],
        
        # ChannelHealthDashboard.tsx - closing paren expected
        "frontend/src/components/Channels/ChannelHealthDashboard.tsx": [
            (r"(map\([^)]*)\s*\{", r"\1) {", "Close map parenthesis"),
        ],
    }
    
    root = Path("C:/Users/Hp/projects/ytempire-mvp")
    
    for file_path, patterns in fixes.items():
        full_path = root / file_path
        if full_path.exists():
            try:
                with open(full_path, 'r', encoding='utf-8') as f:
                    content = f.read()
                
                original = content
                for pattern, replacement, desc in patterns:
                    content = re.sub(pattern, replacement, content, flags=re.MULTILINE)
                
                if content != original:
                    with open(full_path, 'w', encoding='utf-8') as f:
                        f.write(content)
                    print(f"Fixed {full_path.name}")
            except Exception as e:
                print(f"Error in {full_path.name}: {e}")

def fix_property_assignment_errors():
    """Fix property assignment and object literal errors"""
    
    files_to_fix = [
        "frontend/src/components/Analytics/AnalyticsDashboard.tsx",
        "frontend/src/components/Animations/styledComponents.ts",
        "frontend/src/components/Animations/variants.ts",
        "frontend/src/components/BatchOperations/BatchOperations.tsx",
        "frontend/src/components/BulkOperations/EnhancedBulkOperations.tsx",
        "frontend/src/components/Channels/ChannelHealthDashboard.tsx",
    ]
    
    root = Path("C:/Users/Hp/projects/ytempire-mvp")
    
    for file_path in files_to_fix:
        full_path = root / file_path
        if full_path.exists():
            try:
                with open(full_path, 'r', encoding='utf-8') as f:
                    lines = f.readlines()
                
                fixed_lines = []
                for i, line in enumerate(lines):
                    # Fix object property assignment
                    if re.match(r'^\s*\w+\s*:\s*[^,}\n]+$', line) and i + 1 < len(lines):
                        next_line = lines[i + 1]
                        if re.match(r'^\s*\w+\s*:', next_line):
                            line = line.rstrip() + ',\n'
                    
                    # Fix property destructuring
                    line = re.sub(r'\{([^:}]+):([^}]+)\}', r'{ \1: \2 }', line)
                    
                    # Fix object literal syntax
                    line = re.sub(r'(\w+):\s*;', r'\1: null,', line)
                    
                    fixed_lines.append(line)
                
                with open(full_path, 'w', encoding='utf-8') as f:
                    f.writelines(fixed_lines)
                
                print(f"Fixed property assignments in {full_path.name}")
            except Exception as e:
                print(f"Error fixing {full_path.name}: {e}")

def fix_jsx_parent_element_errors():
    """Fix JSX parent element errors"""
    
    files_to_fix = [
        "frontend/src/components/Analytics/AnalyticsDashboard.tsx",
    ]
    
    root = Path("C:/Users/Hp/projects/ytempire-mvp")
    
    for file_path in files_to_fix:
        full_path = root / file_path
        if full_path.exists():
            try:
                with open(full_path, 'r', encoding='utf-8') as f:
                    content = f.read()
                
                # Fix JSX with multiple root elements
                content = re.sub(
                    r'return\s*\(\s*\n\s*(<\w+[^>]*>.*?</\w+>)\s*\n\s*(<\w+)',
                    r'return (\n    <>\n      \1\n      \2',
                    content,
                    flags=re.DOTALL
                )
                
                # Ensure fragments are closed
                if '<>' in content and not '</>' in content:
                    # Find the return statement and add closing fragment
                    content = re.sub(
                        r'(return\s*\(\s*<>.*?)(\s*\)\s*;?\s*})',
                        r'\1\n    </>\2',
                        content,
                        flags=re.DOTALL
                    )
                
                with open(full_path, 'w', encoding='utf-8') as f:
                    f.write(content)
                
                print(f"Fixed JSX parent elements in {full_path.name}")
            except Exception as e:
                print(f"Error fixing {full_path.name}: {e}")

def main():
    print("Fixing specific syntax errors...")
    print("=" * 50)
    
    fix_file_specific_errors()
    fix_property_assignment_errors()
    fix_jsx_parent_element_errors()
    
    print("=" * 50)
    print("Specific fixes applied")

if __name__ == "__main__":
    main()