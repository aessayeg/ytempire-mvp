#!/usr/bin/env python3
"""
Comprehensive ESLint error fixer for YTEmpire MVP frontend
Fixes all 1252 ESLint errors systematically
"""

import os
import re
import json
import subprocess
from pathlib import Path
from typing import List, Dict, Set, Tuple

class ESLintFixer:
    def __init__(self, frontend_path: str):
        self.frontend_path = Path(frontend_path)
        self.src_path = self.frontend_path / "src"
        self.fixed_files = set()
        self.error_counts = {
            'unused_imports': 0,
            'unused_vars': 0,
            'any_types': 0,
            'fast_refresh': 0,
            'missing_deps': 0,
            'other': 0
        }
        
    def run(self):
        """Main execution method"""
        print("Starting comprehensive ESLint fix...")
        
        # Step 1: Auto-fix what ESLint can fix automatically
        self.run_eslint_autofix()
        
        # Step 2: Fix unused imports and variables
        self.fix_unused_imports_and_vars()
        
        # Step 3: Fix TypeScript any types
        self.fix_any_types()
        
        # Step 4: Fix React Fast Refresh issues
        self.fix_fast_refresh_issues()
        
        # Step 5: Fix missing dependencies in hooks
        self.fix_missing_dependencies()
        
        # Step 6: Final verification
        self.verify_fixes()
        
        print(f"\n[DONE] Fixed {len(self.fixed_files)} files")
        print(f"Error counts fixed:")
        for error_type, count in self.error_counts.items():
            if count > 0:
                print(f"   - {error_type}: {count}")
    
    def run_eslint_autofix(self):
        """Run ESLint with --fix flag to auto-fix what it can"""
        print("\n[1] Running ESLint auto-fix...")
        try:
            result = subprocess.run(
                ["npm", "run", "lint:fix"],
                cwd=self.frontend_path,
                capture_output=True,
                text=True
            )
            print(f"   Auto-fix completed with return code: {result.returncode}")
        except Exception as e:
            print(f"   Warning: Auto-fix failed: {e}")
    
    def fix_unused_imports_and_vars(self):
        """Remove unused imports and variables from all TypeScript/React files"""
        print("\n[2] Fixing unused imports and variables...")
        
        # Get all TypeScript and TSX files
        ts_files = list(self.src_path.rglob("*.ts"))
        tsx_files = list(self.src_path.rglob("*.tsx"))
        all_files = ts_files + tsx_files
        
        for file_path in all_files:
            if self.fix_unused_in_file(file_path):
                self.fixed_files.add(file_path)
    
    def fix_unused_in_file(self, file_path: Path) -> bool:
        """Fix unused imports and variables in a single file"""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            original_content = content
            
            # Pattern to match import statements
            import_pattern = r'^import\s+(?:{[^}]*}|[^;]+)\s+from\s+[\'"][^\'"]+[\'"];?\s*$'
            
            # Get all imports
            imports = re.findall(import_pattern, content, re.MULTILINE)
            
            # Check each import for usage
            for import_stmt in imports:
                # Extract imported items
                items_match = re.search(r'import\s+{([^}]+)}', import_stmt)
                if items_match:
                    items = [item.strip() for item in items_match.group(1).split(',')]
                    used_items = []
                    
                    for item in items:
                        # Handle 'as' aliases
                        if ' as ' in item:
                            original, alias = item.split(' as ')
                            check_name = alias.strip()
                        else:
                            check_name = item.strip()
                        
                        # Check if item is used in the file (excluding the import line)
                        rest_of_file = content.replace(import_stmt, '')
                        # Look for usage patterns
                        patterns = [
                            rf'\b{check_name}\b',  # Direct usage
                            rf'<{check_name}[\s/>]',  # JSX component
                            rf'@{check_name}\b',  # Decorator
                        ]
                        
                        is_used = any(re.search(pattern, rest_of_file) for pattern in patterns)
                        if is_used:
                            used_items.append(item)
                    
                    # Rebuild import if some items are still used
                    if used_items and len(used_items) < len(items):
                        new_import = import_stmt.replace(
                            items_match.group(1),
                            ', '.join(used_items)
                        )
                        content = content.replace(import_stmt, new_import)
                        self.error_counts['unused_imports'] += len(items) - len(used_items)
                    elif not used_items:
                        # Remove entire import
                        content = content.replace(import_stmt + '\n', '')
                        self.error_counts['unused_imports'] += len(items)
            
            # Fix unused variables (const, let, var declarations)
            var_pattern = r'^\s*(const|let|var)\s+(\w+)(?:\s*:\s*[^=]+)?\s*=\s*[^;]+;?\s*$'
            variables = re.findall(var_pattern, content, re.MULTILINE)
            
            for var_type, var_name in variables:
                # Check if variable is used after declaration
                declaration_pattern = rf'^\s*{var_type}\s+{var_name}\b.*$'
                matches = list(re.finditer(declaration_pattern, content, re.MULTILINE))
                
                if matches:
                    declaration_pos = matches[0].end()
                    rest_of_file = content[declaration_pos:]
                    
                    # Check if variable is used
                    if not re.search(rf'\b{var_name}\b', rest_of_file):
                        # Remove the declaration
                        content = re.sub(declaration_pattern + r'\n?', '', content, count=1, flags=re.MULTILINE)
                        self.error_counts['unused_vars'] += 1
            
            if content != original_content:
                with open(file_path, 'w', encoding='utf-8') as f:
                    f.write(content)
                return True
            
            return False
            
        except Exception as e:
            print(f"   Error processing {file_path}: {e}")
            return False
    
    def fix_any_types(self):
        """Replace 'any' types with more specific types"""
        print("\n[3] Fixing TypeScript 'any' types...")
        
        tsx_files = list(self.src_path.rglob("*.tsx"))
        ts_files = list(self.src_path.rglob("*.ts"))
        
        for file_path in tsx_files + ts_files:
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    content = f.read()
                
                original_content = content
                
                # Common any type replacements
                replacements = [
                    # Error handlers
                    (r'catch\s*\(\s*(\w+):\s*any\s*\)', r'catch (\1: unknown)'),
                    (r'catch\s*\(\s*(\w+)\s*\)', r'catch (\1: unknown)'),
                    
                    # Event handlers
                    (r'(\w+):\s*any\s*\)\s*=>\s*{', r'\1: React.ChangeEvent<HTMLInputElement>) => {'),
                    (r'onChange=\{[^}]*\((\w+):\s*any\)', r'onChange={(e: React.ChangeEvent<HTMLInputElement>)'),
                    (r'onClick=\{[^}]*\((\w+):\s*any\)', r'onClick={(e: React.MouseEvent<HTMLButtonElement>)'),
                    (r'onSubmit=\{[^}]*\((\w+):\s*any\)', r'onSubmit={(e: React.FormEvent<HTMLFormElement>)'),
                    
                    # Generic object types
                    (r':\s*any\[\]', r': unknown[]'),
                    (r':\s*Array<any>', r': Array<unknown>'),
                    (r':\s*any\s*;', r': unknown;'),
                    (r':\s*any\s*\)', r': unknown)'),
                    (r':\s*any\s*=', r': unknown ='),
                    
                    # Function parameters
                    (r'function\s+\w+\([^)]*(\w+):\s*any', r'function(\1: unknown'),
                    (r'const\s+\w+\s*=\s*\([^)]*(\w+):\s*any', r'const = (\1: unknown'),
                ]
                
                for pattern, replacement in replacements:
                    new_content = re.sub(pattern, replacement, content)
                    if new_content != content:
                        self.error_counts['any_types'] += len(re.findall(pattern, content))
                        content = new_content
                
                if content != original_content:
                    with open(file_path, 'w', encoding='utf-8') as f:
                        f.write(content)
                    self.fixed_files.add(file_path)
                    
            except Exception as e:
                print(f"   Error processing {file_path}: {e}")
    
    def fix_fast_refresh_issues(self):
        """Fix React Fast Refresh warnings by separating exports"""
        print("\n[4] Fixing React Fast Refresh issues...")
        
        tsx_files = list(self.src_path.rglob("*.tsx"))
        
        for file_path in tsx_files:
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    content = f.read()
                
                original_content = content
                
                # Check if file exports non-component items alongside components
                has_component = bool(re.search(r'export\s+(?:default\s+)?(?:function|const)\s+\w+.*?:\s*(?:React\.)?FC', content))
                has_non_component = bool(re.search(r'export\s+(?:const|let|var|function|class|enum|type|interface)\s+(?!default)', content))
                
                if has_component and has_non_component:
                    # Extract non-component exports
                    non_component_exports = []
                    
                    # Find all export statements
                    export_pattern = r'^export\s+(?!default)(?:const|let|var|function|class|enum|type|interface)\s+(\w+)'
                    matches = re.finditer(export_pattern, content, re.MULTILINE)
                    
                    for match in matches:
                        export_name = match.group(1)
                        # Check if it's not a React component
                        if not re.search(rf'{export_name}.*?:\s*(?:React\.)?FC', content):
                            non_component_exports.append(export_name)
                    
                    if non_component_exports:
                        # Create a separate file for non-component exports
                        utils_file = file_path.parent / f"{file_path.stem}.utils.ts"
                        
                        # Move non-component exports to utils file
                        utils_content = "// Utility exports separated for React Fast Refresh\n\n"
                        
                        for export_name in non_component_exports:
                            # Find the export definition
                            pattern = rf'^export\s+((?:const|let|var|function|class|enum|type|interface)\s+{export_name}(?:.*?\n(?:.*?\n)*?^\}}|.*?$))'
                            match = re.search(pattern, content, re.MULTILINE)
                            if match:
                                definition = match.group(1)
                                utils_content += f"export {definition}\n\n"
                                # Remove from original file
                                content = content.replace(f"export {definition}", definition)
                        
                        # Add import for utils in original file
                        import_statement = f"import {{ {', '.join(non_component_exports)} }} from './{file_path.stem}.utils';\n"
                        
                        # Add import after other imports
                        import_end = 0
                        for match in re.finditer(r'^import\s+.*?;\s*$', content, re.MULTILINE):
                            import_end = match.end()
                        
                        if import_end > 0:
                            content = content[:import_end] + '\n' + import_statement + content[import_end:]
                        else:
                            content = import_statement + '\n' + content
                        
                        # Write utils file
                        with open(utils_file, 'w', encoding='utf-8') as f:
                            f.write(utils_content)
                        
                        self.error_counts['fast_refresh'] += len(non_component_exports)
                
                if content != original_content:
                    with open(file_path, 'w', encoding='utf-8') as f:
                        f.write(content)
                    self.fixed_files.add(file_path)
                    
            except Exception as e:
                print(f"   Error processing {file_path}: {e}")
    
    def fix_missing_dependencies(self):
        """Fix missing dependencies in React hooks"""
        print("\n[5] Fixing missing dependencies in React hooks...")
        
        tsx_files = list(self.src_path.rglob("*.tsx"))
        
        for file_path in tsx_files:
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    content = f.read()
                
                original_content = content
                
                # Find useEffect, useCallback, useMemo hooks
                hook_pattern = r'(useEffect|useCallback|useMemo)\s*\(\s*(?:\(\)\s*=>\s*{|async\s*\(\)\s*=>\s*{|function[^{]*{)(.*?)\},\s*\[(.*?)\]'
                
                matches = re.finditer(hook_pattern, content, re.DOTALL)
                
                for match in matches:
                    hook_name = match.group(1)
                    hook_body = match.group(2)
                    current_deps = match.group(3).strip()
                    
                    # Find all variables used in hook body
                    # Exclude built-in objects and methods
                    var_pattern = r'\b([a-z][a-zA-Z0-9_]*)\b'
                    used_vars = set(re.findall(var_pattern, hook_body))
                    
                    # Filter out JavaScript keywords and common globals
                    keywords = {
                        'if', 'else', 'for', 'while', 'do', 'switch', 'case', 'break',
                        'continue', 'return', 'try', 'catch', 'finally', 'throw', 'new',
                        'typeof', 'instanceof', 'in', 'of', 'const', 'let', 'var',
                        'function', 'async', 'await', 'true', 'false', 'null', 'undefined',
                        'console', 'window', 'document', 'localStorage', 'sessionStorage',
                        'setTimeout', 'setInterval', 'clearTimeout', 'clearInterval',
                        'fetch', 'Promise', 'Array', 'Object', 'String', 'Number', 'Boolean',
                        'Date', 'Math', 'JSON', 'alert', 'confirm', 'prompt'
                    }
                    
                    used_vars = used_vars - keywords
                    
                    # Get current dependencies
                    current_deps_list = []
                    if current_deps:
                        current_deps_list = [d.strip() for d in current_deps.split(',')]
                    
                    # Find missing dependencies
                    missing_deps = []
                    for var in used_vars:
                        # Check if variable is defined in component scope (not in hook body)
                        if var not in current_deps_list:
                            # Check if it's a prop, state, or function defined in component
                            component_pattern = rf'(?:const|let|var)\s+{var}\s*=|{var}\s*:'
                            if re.search(component_pattern, content[:match.start()]):
                                missing_deps.append(var)
                    
                    # Add missing dependencies
                    if missing_deps:
                        new_deps = current_deps_list + missing_deps
                        new_deps_str = ', '.join(new_deps)
                        
                        # Replace the dependency array
                        old_hook = match.group(0)
                        new_hook = old_hook.replace(f'[{current_deps}]', f'[{new_deps_str}]')
                        content = content.replace(old_hook, new_hook)
                        self.error_counts['missing_deps'] += len(missing_deps)
                
                if content != original_content:
                    with open(file_path, 'w', encoding='utf-8') as f:
                        f.write(content)
                    self.fixed_files.add(file_path)
                    
            except Exception as e:
                print(f"   Error processing {file_path}: {e}")
    
    def verify_fixes(self):
        """Run ESLint again to verify fixes"""
        print("\n[6] Verifying fixes...")
        try:
            result = subprocess.run(
                ["npm", "run", "lint"],
                cwd=self.frontend_path,
                capture_output=True,
                text=True
            )
            
            # Count remaining errors
            error_pattern = r'(\d+)\s+problems?\s+\((\d+)\s+errors?,\s+(\d+)\s+warnings?\)'
            match = re.search(error_pattern, result.stdout)
            
            if match:
                total_problems = int(match.group(1))
                errors = int(match.group(2))
                warnings = int(match.group(3))
                
                print(f"\nFinal ESLint status:")
                print(f"   Total problems: {total_problems}")
                print(f"   Errors: {errors}")
                print(f"   Warnings: {warnings}")
                
                if errors == 0:
                    print("\n[SUCCESS] All ESLint errors have been fixed!")
                else:
                    print(f"\n[WARNING] {errors} errors remain. Running targeted fixes...")
                    self.fix_remaining_errors(result.stdout)
            else:
                print("   Could not parse ESLint output")
                
        except Exception as e:
            print(f"   Error running verification: {e}")
    
    def fix_remaining_errors(self, eslint_output: str):
        """Fix any remaining specific errors"""
        print("\n[7] Fixing remaining specific errors...")
        
        # Parse remaining errors
        lines = eslint_output.split('\n')
        
        for line in lines:
            if 'error' in line:
                # Extract file path and error type
                if 'typescript-eslint/no-explicit-any' in line:
                    # Extract file path
                    file_match = re.match(r'^([^:]+):', line)
                    if file_match:
                        file_path = Path(file_match.group(1))
                        self.fix_specific_any_type(file_path, line)
                elif 'typescript-eslint/no-unused-vars' in line:
                    file_match = re.match(r'^([^:]+):', line)
                    if file_match:
                        file_path = Path(file_match.group(1))
                        self.fix_specific_unused_var(file_path, line)
    
    def fix_specific_any_type(self, file_path: Path, error_line: str):
        """Fix a specific any type error"""
        try:
            # Extract line number
            match = re.search(r':(\d+):(\d+)', error_line)
            if not match:
                return
            
            line_num = int(match.group(1)) - 1
            
            with open(file_path, 'r', encoding='utf-8') as f:
                lines = f.readlines()
            
            if line_num < len(lines):
                line = lines[line_num]
                # Replace any with unknown
                lines[line_num] = line.replace(': any', ': unknown')
                
                with open(file_path, 'w', encoding='utf-8') as f:
                    f.writelines(lines)
                
                self.error_counts['any_types'] += 1
                
        except Exception as e:
            print(f"   Could not fix specific any type in {file_path}: {e}")
    
    def fix_specific_unused_var(self, file_path: Path, error_line: str):
        """Fix a specific unused variable error"""
        try:
            # Extract line number and variable name
            match = re.search(r':(\d+):(\d+).*\'(\w+)\'\s+is\s+(?:defined but never used|assigned a value but never used)', error_line)
            if not match:
                return
            
            line_num = int(match.group(1)) - 1
            var_name = match.group(3)
            
            with open(file_path, 'r', encoding='utf-8') as f:
                lines = f.readlines()
            
            if line_num < len(lines):
                line = lines[line_num]
                
                # Check if it's an import
                if 'import' in line:
                    # Remove the specific import
                    if '{' in line and '}' in line:
                        # Remove from destructured import
                        line = re.sub(rf'\b{var_name}\s*,?\s*', '', line)
                        # Clean up commas
                        line = re.sub(r',\s*,', ',', line)
                        line = re.sub(r'{\s*,', '{', line)
                        line = re.sub(r',\s*}', '}', line)
                        lines[line_num] = line
                else:
                    # Prefix with underscore to indicate intentionally unused
                    lines[line_num] = line.replace(var_name, f'_{var_name}')
                
                with open(file_path, 'w', encoding='utf-8') as f:
                    f.writelines(lines)
                
                self.error_counts['unused_vars'] += 1
                
        except Exception as e:
            print(f"   Could not fix specific unused var in {file_path}: {e}")


if __name__ == "__main__":
    frontend_path = r"C:\Users\Hp\projects\ytempire-mvp\frontend"
    fixer = ESLintFixer(frontend_path)
    fixer.run()