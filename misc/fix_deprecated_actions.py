#!/usr/bin/env python3
"""
Fix deprecated GitHub Actions versions in workflow files
"""
import os
import re
from pathlib import Path

# Define the replacements
REPLACEMENTS = {
    # Upgrade deprecated v2 and v3 versions to latest
    r'uses: actions/upload-artifact@v[23]': 'uses: actions/upload-artifact@v4',
    r'uses: actions/download-artifact@v[23]': 'uses: actions/download-artifact@v4',
    r'uses: actions/cache@v[23]': 'uses: actions/cache@v4',
    r'uses: codecov/codecov-action@v[23]': 'uses: codecov/codecov-action@v4',
    r'uses: 8398a7/action-slack@v[23]': 'uses: 8398a7/action-slack@v4',
    r'uses: github/codeql-action/upload-sarif@v2': 'uses: github/codeql-action/upload-sarif@v3',
    r'uses: returntocorp/semgrep-action@v1': 'uses: returntocorp/semgrep-action@v2',
    r'uses: fossa-contrib/fossa-action@v2': 'uses: fossa-contrib/fossa-action@v3',
    r'uses: gitleaks/gitleaks-action@v2': 'uses: gitleaks/gitleaks-action@v2',  # Keep v2 as it's latest
    r'uses: anchore/scan-action@v3': 'uses: anchore/scan-action@v4',
}

def fix_workflow_file(filepath):
    """Fix deprecated actions in a workflow file"""
    with open(filepath, 'r', encoding='utf-8') as f:
        content = f.read()
    
    original_content = content
    changes_made = []
    
    for pattern, replacement in REPLACEMENTS.items():
        matches = re.findall(pattern, content)
        if matches:
            content = re.sub(pattern, replacement, content)
            changes_made.append(f"  - {matches[0]} -> {replacement}")
    
    if content != original_content:
        with open(filepath, 'w', encoding='utf-8') as f:
            f.write(content)
        return changes_made
    return []

def main():
    """Main function to fix all workflow files"""
    workflows_dir = Path(r"C:\Users\Hp\projects\ytempire-mvp\.github\workflows")
    
    if not workflows_dir.exists():
        print(f"Workflows directory not found: {workflows_dir}")
        return
    
    print("Fixing deprecated GitHub Actions in workflow files...")
    print("=" * 60)
    
    total_changes = 0
    for workflow_file in workflows_dir.glob("*.yml"):
        changes = fix_workflow_file(workflow_file)
        if changes:
            print(f"\n{workflow_file.name}:")
            for change in changes:
                print(change)
            total_changes += len(changes)
    
    print("\n" + "=" * 60)
    print(f"Total changes made: {total_changes}")
    
    if total_changes > 0:
        print("\n✅ Successfully updated all deprecated actions!")
    else:
        print("\n✅ No deprecated actions found - all workflows are up to date!")

if __name__ == "__main__":
    main()