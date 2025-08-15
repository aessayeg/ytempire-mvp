#!/usr/bin/env python3
"""
Final comprehensive syntax fix for TypeScript/JSX files
"""

import os
import re
from pathlib import Path
from typing import Dict, List, Tuple

def fix_interface_syntax(content: str) -> str:
    """Fix interface and type declaration syntax"""
    
    # Fix interfaces with incorrect punctuation
    # Remove semicolons and commas after property names in interfaces
    lines = content.split('\n')
    fixed_lines = []
    in_interface = False
    in_type = False
    
    for line in lines:
        # Detect interface/type start
        if re.match(r'^\s*(interface|type)\s+\w+\s*({|=)', line):
            in_interface = 'interface' in line
            in_type = 'type' in line
            fixed_lines.append(line)
            continue
        
        # Detect end of interface/type
        if (in_interface or in_type) and re.match(r'^\s*}', line):
            in_interface = False
            in_type = False
            fixed_lines.append(line)
            continue
        
        # Fix property declarations in interfaces
        if in_interface or in_type:
            # Remove extra semicolons and commas in property names
            line = re.sub(r'(\w+)\s*[,;]\s*:', r'\1:', line)
            # Fix property endings - should be semicolon or comma
            line = re.sub(r'(\w+:\s*[^;,}\n]+)[,;]\s*,', r'\1,', line)
            # Remove double punctuation
            line = re.sub(r'[,;]\s*[,;]', ';', line)
        
        fixed_lines.append(line)
    
    return '\n'.join(fixed_lines)

def fix_import_statements(content: str) -> str:
    """Fix import statement syntax"""
    
    # Fix malformed imports
    content = re.sub(r'import\s*\n\s*from', 'import {} from', content)
    content = re.sub(r'import\s+from\s+', 'import {} from ', content)
    
    # Fix missing from keyword
    content = re.sub(r"import\s+{([^}]+)}\s*'", r"import { \1 } from '", content)
    
    return content

def fix_object_literals(content: str) -> str:
    """Fix object literal syntax"""
    
    # Remove commas and semicolons in property names
    content = re.sub(r'(\w+)\s*,\s*:', r'\1:', content)
    
    # Fix empty property values
    content = re.sub(r':\s*,', r': null,', content)
    content = re.sub(r':\s*}', r': null }', content)
    
    return content

def fix_jsx_syntax(content: str) -> str:
    """Fix JSX specific syntax issues"""
    
    # Fix JSX fragment issues
    content = re.sub(r'</>([^<]*)<>', r'</>\1<>', content)
    
    # Fix unclosed JSX elements in conditionals
    content = re.sub(r'(\?\s*<[^>]+>)([^<]*)(:\s*<)', r'\1\2</>\3', content)
    
    return content

def fix_specific_files():
    """Fix specific known problematic files"""
    
    fixes = {
        "frontend/src/components/Analytics/CompetitiveAnalysisDashboard.tsx": {
            "patterns": [
                # Fix interface properties with incorrect punctuation
                (r'channelName: string;,', 'channelName: string;'),
                (r'channelId: string,', 'channelId: string;'),
                (r'thumbnailUrl: string;,', 'thumbnailUrl: string;'),
                (r'subscriberCount: number,', 'subscriberCount: number;'),
                (r'videoCount: number;,', 'videoCount: number;'),
                (r'viewCount: number,', 'viewCount: number;'),
                (r'category: string;,', 'category: string;'),
                (r'country: string,', 'country: string;'),
                (r'joinedDate: Date;,', 'joinedDate: Date;'),
                (r'isTracking: boolean,', 'isTracking: boolean;'),
                (r'lastUpdated: Date}', 'lastUpdated: Date;\n}'),
                # Fix other interface issues
                (r'avgViews: number;,', 'avgViews: number;'),
                (r'avgLikes: number,', 'avgLikes: number;'),
                (r'avgComments: number;,', 'avgComments: number;'),
                (r'engagementRate: number,', 'engagementRate: number;'),
                (r'uploadFrequency: number; // videos per week,', 'uploadFrequency: number; // videos per week'),
                (r'estimatedRevenue: number,', 'estimatedRevenue: number;'),
                (r'growthRate: number; // % per month,', 'growthRate: number; // % per month'),
                (r'contentQualityScore: number; // 0-100,', 'contentQualityScore: number; // 0-100'),
                (r'audienceRetention: number; // percentage,', 'audienceRetention: number; // percentage'),
                (r'clickThroughRate: number}', 'clickThroughRate: number;\n}'),
                # Fix MarketInsight interface
                (r"trend: string,", "trend: string;"),
                (r"impact: 'high' \| 'medium' \| 'low';,", "impact: 'high' | 'medium' | 'low';"),
                (r"description: string,", "description: string;"),
                (r"recommendedAction: string}", "recommendedAction: string;\n}"),
                # Fix ContentGap interface
                (r"topic: string,", "topic: string;"),
                (r"competitorsCovering: number;,", "competitorsCovering: number;"),
                (r"potentialViews: number,", "potentialViews: number;"),
                (r"difficulty: 'easy' \| 'medium' \| 'hard';,", "difficulty: 'easy' | 'medium' | 'hard';"),
                (r"recommendedApproach: string}", "recommendedApproach: string;\n}"),
                # Fix JSX issues
                (r'onChange=\{.*?\(.*?\)\{.*?\}', 'onChange={() => toggleCompetitorSelection(competitor.id)}'),
                (r'const comp = competitors\.find\(c => c\.id === compId</>', 'const comp = competitors.find(c => c.id === compId)'),
                (r'\n\s*</>', ''),  # Remove stray closing fragments
                (r'</>(\s*\);)', r'\1'),  # Remove fragment before closing paren
            ]
        },
        "frontend/src/components/Auth/TwoFactorAuth.tsx": {
            "patterns": [
                (r"import\s*\n\s*from", "import { useState, useEffect } from"),
            ]
        },
        "frontend/src/components/Animations/styledComponents.ts": {
            "patterns": [
                (r'styled\.\w+`([^`]*);`', r'styled.\1`\1`'),
            ]
        },
        "frontend/src/components/Analytics/AnalyticsDashboard.tsx": {
            "patterns": [
                # Fix interface issues
                (r'(\w+),\s*:', r'\1:'),
            ]
        },
        "frontend/src/components/BatchOperations/BatchOperations.tsx": {
            "patterns": [
                # Fix interface properties
                (r'(\w+),\s*:', r'\1:'),
            ]
        },
        "frontend/src/components/BulkOperations/EnhancedBulkOperations.tsx": {
            "patterns": [
                # Fix interface properties
                (r'(\w+),\s*:', r'\1:'),
            ]
        },
        "frontend/src/components/Accessibility/announcementManager.ts": {
            "patterns": [
                # Fix interface properties
                (r'(\w+),\s*:', r'\1:'),
            ]
        }
    }
    
    root = Path("C:/Users/Hp/projects/ytempire-mvp")
    
    for file_path, fix_config in fixes.items():
        full_path = root / file_path
        if full_path.exists():
            try:
                with open(full_path, 'r', encoding='utf-8') as f:
                    content = f.read()
                
                original = content
                for pattern, replacement in fix_config["patterns"]:
                    content = re.sub(pattern, replacement, content)
                
                if content != original:
                    with open(full_path, 'w', encoding='utf-8') as f:
                        f.write(content)
                    print(f"Fixed: {full_path.name}")
            except Exception as e:
                print(f"Error fixing {full_path.name}: {e}")

def fix_all_files():
    """Apply general fixes to all TypeScript/TSX files"""
    
    frontend_src = Path("C:/Users/Hp/projects/ytempire-mvp/frontend/src")
    
    for file_path in list(frontend_src.rglob("*.tsx")) + list(frontend_src.rglob("*.ts")):
        if file_path.name.endswith('.d.ts'):
            continue
        
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            original = content
            
            # Apply general fixes
            content = fix_interface_syntax(content)
            content = fix_import_statements(content)
            content = fix_object_literals(content)
            content = fix_jsx_syntax(content)
            
            if content != original:
                with open(file_path, 'w', encoding='utf-8') as f:
                    f.write(content)
                print(f"Fixed: {file_path.name}")
        
        except Exception as e:
            continue

def main():
    print("Applying final syntax fixes...")
    print("=" * 50)
    
    # First fix specific known issues
    fix_specific_files()
    
    # Then apply general fixes
    fix_all_files()
    
    print("=" * 50)
    print("Syntax fixes complete")

if __name__ == "__main__":
    main()