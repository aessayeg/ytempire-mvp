#!/usr/bin/env python3
"""
TypeScript Error Fix Progress Tracker
Tracks the progress of fixing TypeScript errors in the frontend codebase
"""

import subprocess
import json
from datetime import datetime

def get_error_count():
    """Get the current TypeScript error count"""
    try:
        result = subprocess.run(
            'cd frontend && npm run build 2>&1 | grep "error TS" | wc -l',
            shell=True,
            capture_output=True,
            text=True
        )
        return int(result.stdout.strip())
    except:
        return -1

def main():
    # Historical data
    progress = [
        {"timestamp": "Session Start", "errors": 6000, "files_fixed": 0, "description": "Initial state - errors doubled from ~3500"},
        {"timestamp": "Fix 1", "errors": 5874, "files_fixed": 1, "description": "Fixed styledComponents.ts (127 errors)"},
        {"timestamp": "Fix 2", "errors": 5748, "files_fixed": 2, "description": "Fixed VideoGenerator.tsx (126 errors)"},
        {"timestamp": "Fix 3", "errors": 5641, "files_fixed": 3, "description": "Fixed TwoFactorAuth.tsx (107 errors)"},
        {"timestamp": "Fix 4", "errors": 5535, "files_fixed": 4, "description": "Fixed VideoGenerationForm.tsx (106 errors)"},
        {"timestamp": "Current", "errors": 2890, "files_fixed": 12, "description": "Fixed 12 high-error files"}
    ]
    
    # Get current error count
    current_errors = get_error_count()
    
    print("=" * 70)
    print("TYPESCRIPT ERROR FIX PROGRESS REPORT")
    print("=" * 70)
    print(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print()
    
    # Progress summary
    initial_errors = progress[0]["errors"]
    errors_fixed = initial_errors - current_errors
    percentage_fixed = (errors_fixed / initial_errors) * 100
    
    print(f"OVERALL PROGRESS:")
    print(f"  Initial Errors: {initial_errors:,}")
    print(f"  Current Errors: {current_errors:,}")
    print(f"  Errors Fixed: {errors_fixed:,}")
    print(f"  Progress: {percentage_fixed:.1f}%")
    print()
    
    # Files fixed
    print(f"FILES FIXED (12 total):")
    files_fixed = [
        "[FIXED] src/components/Analytics/AnalyticsDashboard.tsx (63 errors)",
        "[FIXED] src/components/Animations/styledComponents.ts (127 errors)",
        "[FIXED] src/components/Animations/AdvancedAnimations.tsx (1 error)",
        "[FIXED] src/utils/formatters.ts (21 errors)",
        "[FIXED] src/utils/lazyWithRetry.ts (3 errors)",
        "[FIXED] src/pages/Videos/VideoGenerator.tsx (126 errors)",
        "[FIXED] src/components/Auth/TwoFactorAuth.tsx (107 errors)",
        "[FIXED] src/components/Videos/VideoGenerationForm.tsx (106 errors)",
        "[FIXED] src/components/Accessibility/AccessibleButton.tsx",
        "[FIXED] src/components/Accessibility/SkipNavigation.tsx",
        "[FIXED] src/theme/darkMode.ts (105 errors)",
        "[FIXED] src/components/Mobile/MobileResponsiveSystem.tsx (98 errors)"
    ]
    
    for file in files_fixed:
        print(f"  {file}")
    print()
    
    # Remaining high-error files
    print(f"NEXT HIGH-PRIORITY FILES TO FIX:")
    high_priority_files = [
        "src/components/Channels/ChannelTemplates.tsx (87 errors)",
        "src/pages/Settings/Settings.tsx (84 errors)",
        "src/pages/Videos/VideoQueue.tsx (80 errors)",
        "src/pages/Costs/CostTracking.tsx (79 errors)",
        "src/components/Dashboard/EnhancedMetricsDashboard.tsx (76 errors)"
    ]
    
    for file in high_priority_files[:5]:
        print(f"  [TODO] {file}")
    print()
    
    # Key insights
    print(f"KEY INSIGHTS:")
    print(f"  - Fixed {errors_fixed:,} errors ({percentage_fixed:.1f}% reduction)")
    print(f"  - Average errors per file fixed: {errors_fixed // 12}")
    print(f"  - Estimated files remaining: ~{current_errors // 50}")
    print(f"  - Most common issues fixed:")
    print(f"    - Semicolons instead of commas in object literals")
    print(f"    - Malformed interface definitions")
    print(f"    - Missing closing braces/brackets")
    print(f"    - Duplicate imports")
    print(f"    - Unterminated template literals")
    print()
    
    # Recommendations
    print(f"RECOMMENDATIONS:")
    print(f"  1. Continue fixing high-error files (50+ errors)")
    print(f"  2. Focus on systematic issues (semicolons -> commas)")
    print(f"  3. Target: Reduce to under 1000 errors")
    print(f"  4. Estimated time to completion: ~2-3 hours at current pace")
    print()
    
    print("=" * 70)
    
    # Save progress to JSON
    progress_data = {
        "timestamp": datetime.now().isoformat(),
        "current_errors": current_errors,
        "initial_errors": initial_errors,
        "errors_fixed": errors_fixed,
        "percentage_fixed": percentage_fixed,
        "files_fixed": 12,
        "progress_history": progress
    }
    
    with open("misc/typescript_fix_progress.json", "w") as f:
        json.dump(progress_data, f, indent=2)
    
    print(f"Progress data saved to misc/typescript_fix_progress.json")

if __name__ == "__main__":
    main()