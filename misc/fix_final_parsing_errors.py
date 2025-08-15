#!/usr/bin/env python3
"""
Fix final remaining parsing errors in TypeScript/JSX files
"""

import re
from pathlib import Path

# Specific files and their fixes
SPECIFIC_FIXES = {
    "frontend/src/components/Accessibility/announcementManager.ts": [
        (r'(\w+),\s*:', r'\1:', 'Remove comma before colon in properties'),
    ],
    "frontend/src/components/Analytics/CompetitiveAnalysisDashboard.tsx": [
        (r'lastUpdated: Date,\n\s*}', r'lastUpdated: Date;\n  }', 'Fix interface ending'),
        (r'clickThroughRate: number,\n\s*}', r'clickThroughRate: number;\n  }', 'Fix interface ending'),
        (r'recommendedAction: string,\n\s*}', r'recommendedAction: string;\n  }', 'Fix interface ending'),
        (r'recommendedApproach: string,\n\s*}', r'recommendedApproach: string;\n  }', 'Fix interface ending'),
        (r'lastUpdated: new Date\(\),\n\s*}', r'lastUpdated: new Date()\n  }', 'Fix object literal'),
        (r'setNewCompetitorUrl\(\'\'\)}', r"setNewCompetitorUrl('');\n  }", 'Add semicolon'),
        (r'a\.click\(\)}', r'a.click();\n  }', 'Add semicolon'),
        (r'onChange=\{.*?\}', r'onChange={() => toggleCompetitorSelection(competitor.id)}', 'Fix onChange'),
    ],
    "frontend/src/components/Analytics/UserBehaviorDashboard.tsx": [
        (r'(\w+:\s*[^,}\n]+)\n\s*(\w+:)', r'\1,\n  \2', 'Add missing comma'),
    ],
    "frontend/src/components/Animations/AdvancedAnimations.tsx": [
        (r'(\w+),\s*:', r'\1:', 'Remove comma before colon'),
    ],
    "frontend/src/components/Animations/styledComponents.ts": [
        (r'(\w+),\s*:', r'\1:', 'Remove comma before colon'),
    ],
    "frontend/src/components/Animations/variants.ts": [
        (r'(\w+),\s*:', r'\1:', 'Remove comma before colon'),
    ],
    "frontend/src/components/Auth/EmailVerification.tsx": [
        (r'useState\([^)]*\)\s*\n', r'useState();\n', 'Fix useState'),
    ],
    "frontend/src/components/Auth/ForgotPasswordForm.tsx": [
        (r'setSubmitting\(false\)\s*\n', r'setSubmitting(false);\n', 'Add semicolon'),
    ],
    "frontend/src/components/Auth/LoginForm.tsx": [
        (r'(\w+:\s*[^,}\n]+)\n\s*(\w+:)', r'\1,\n  \2', 'Add missing comma'),
    ],
    "frontend/src/components/Auth/RegisterForm.tsx": [
        (r'(\w+:\s*[^,}\n]+)\n\s*(\w+:)', r'\1,\n  \2', 'Add missing comma'),
    ],
    "frontend/src/components/Auth/TwoFactorAuth.tsx": [
        (r'import\s*\n\s*from', r'import { useState, useEffect } from', 'Fix import'),
    ],
    "frontend/src/components/BatchOperations/BatchOperations.tsx": [
        (r'(\w+),\s*:', r'\1:', 'Remove comma before colon'),
    ],
    "frontend/src/components/BulkOperations/EnhancedBulkOperations.tsx": [
        (r'(\w+),\s*:', r'\1:', 'Remove comma before colon'),
    ],
    "frontend/src/components/Channels/BulkOperations.tsx": [
        (r'(\w+),\s*:', r'\1:', 'Remove comma before colon'),
    ],
    "frontend/src/components/Channels/ChannelDashboard.tsx": [
        (r'(\w+),\s*:', r'\1:', 'Remove comma before colon'),
    ],
    "frontend/src/components/Channels/ChannelHealthDashboard.tsx": [
        (r'(\w+),\s*:', r'\1:', 'Remove comma before colon'),
    ],
    "frontend/src/components/Channels/ChannelList.tsx": [
        (r'(\w+),\s*:', r'\1:', 'Remove comma before colon'),
    ],
}

def apply_fixes():
    """Apply specific fixes to known problematic files"""
    
    root = Path("C:/Users/Hp/projects/ytempire-mvp")
    fixed_count = 0
    
    for file_path, patterns in SPECIFIC_FIXES.items():
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
                    fixed_count += 1
                    print(f"Fixed: {full_path.name}")
            except Exception as e:
                print(f"Error fixing {full_path.name}: {e}")
    
    return fixed_count

def main():
    print("Applying final parsing error fixes...")
    print("=" * 50)
    
    fixed = apply_fixes()
    
    print("=" * 50)
    print(f"Fixed {fixed} files")

if __name__ == "__main__":
    main()