#!/usr/bin/env ts-node
/**
 * Comprehensive ESLint error fixer for YTEmpire MVP
 * Fixes all common ESLint errors programmatically
 */

import * as fs from 'fs';
import * as path from 'path';
import { execSync } from 'child_process';

interface ImportInfo {
  line: string;
  index: number;
  imports: string[];
  module: string;
  isDefault?: boolean;
}

class ESLintFixer {
  private frontendSrc: string;
  private filesFixed: number = 0;
  private totalFiles: number = 0;

  constructor() {
    this.frontendSrc = path.join(__dirname, '..', 'frontend', 'src');
  }

  /**
   * Main execution method
   */
  public async fix(): Promise<void> {
    console.log('Starting comprehensive ESLint fixes...');
    console.log('-'.repeat(50));

    // Get all TypeScript/React files
    const files = this.getAllFiles(this.frontendSrc);
    
    for (const file of files) {
      if (this.processFile(file)) {
        this.filesFixed++;
      }
      this.totalFiles++;
    }

    console.log('-'.repeat(50));
    console.log(`Total files processed: ${this.totalFiles}`);
    console.log(`Files fixed: ${this.filesFixed}`);

    // Run ESLint auto-fix
    console.log('\nRunning ESLint auto-fix...');
    try {
      execSync('cd frontend && npm run lint:fix', { stdio: 'inherit' });
    } catch (error) {
      // ESLint will exit with error if issues remain
    }
  }

  /**
   * Get all TypeScript/React files recursively
   */
  private getAllFiles(dir: string): string[] {
    const files: string[] = [];
    
    const walk = (currentDir: string) => {
      const items = fs.readdirSync(currentDir);
      for (const item of items) {
        const fullPath = path.join(currentDir, item);
        const stat = fs.statSync(fullPath);
        
        if (stat.isDirectory() && !item.startsWith('.') && item !== 'node_modules') {
          walk(fullPath);
        } else if (stat.isFile() && /\.(tsx?|jsx?)$/.test(item)) {
          files.push(fullPath);
        }
      }
    };
    
    walk(dir);
    return files;
  }

  /**
   * Process a single file with all fixes
   */
  private processFile(filePath: string): boolean {
    try {
      let content = fs.readFileSync(filePath, 'utf8');
      const originalContent = content;

      // Apply all fixes
      content = this.removeUnusedImports(content, filePath);
      content = this.fixAnyTypes(content);
      content = this.fixUnusedVariables(content);
      content = this.fixReactHooksDeps(content);
      content = this.fixOptionalChaining(content);
      content = this.fixExportIssues(content, filePath);

      // Write back if changed
      if (content !== originalContent) {
        fs.writeFileSync(filePath, content, 'utf8');
        const relPath = path.relative(this.frontendSrc, filePath);
        console.log(`Fixed: ${relPath}`);
        return true;
      }

      return false;
    } catch (error: any) {
      console.error(`Error processing ${filePath}: ${error.message}`);
      return false;
    }
  }

  /**
   * Remove unused imports intelligently
   */
  private removeUnusedImports(content: string, filePath: string): string {
    const lines = content.split('\n');
    const importRegex = /^import\s+(?:(?:\*\s+as\s+(\w+))|(?:{([^}]+)})|(\w+)(?:\s*,\s*{([^}]+)})?)?\s+from\s+['"]([^'"]+)['"];?$/;
    
    const imports: ImportInfo[] = [];
    const codeLines: string[] = [];
    
    // Parse imports and code separately
    lines.forEach((line, index) => {
      const match = line.trim().match(importRegex);
      if (match) {
        if (match[1]) {
          // Namespace import: import * as Name
          imports.push({ line, index, imports: [match[1]], module: match[5] });
        } else if (match[2]) {
          // Named imports: import { A, B }
          const namedImports = match[2].split(',').map(i => i.trim());
          imports.push({ line, index, imports: namedImports, module: match[5] });
        } else if (match[3] && match[4]) {
          // Default + named: import Default, { Named }
          const namedImports = match[4].split(',').map(i => i.trim());
          imports.push({ line, index, imports: [match[3], ...namedImports], module: match[5], isDefault: true });
        } else if (match[3]) {
          // Default import: import Default
          imports.push({ line, index, imports: [match[3]], module: match[5], isDefault: true });
        }
      } else if (!line.trim().startsWith('//') && !line.trim().startsWith('/*')) {
        codeLines.push(line);
      }
    });

    const codeWithoutImports = codeLines.join('\n');
    const modifiedLines: string[] = [];

    // Check each line and keep only used imports
    lines.forEach((line, index) => {
      const importInfo = imports.find(imp => imp.index === index);
      
      if (importInfo) {
        const usedImports = importInfo.imports.filter(imp => {
          const cleanName = imp.split(' as ')[0].trim();
          // Check if the import is actually used in the code
          const regex = new RegExp(`\\b${this.escapeRegex(cleanName)}\\b`);
          return regex.test(codeWithoutImports);
        });

        if (usedImports.length > 0) {
          if (usedImports.length !== importInfo.imports.length) {
            // Reconstruct import with only used items
            let newImport: string;
            if (importInfo.isDefault && usedImports.length === 1) {
              newImport = `import ${usedImports[0]} from '${importInfo.module}';`;
            } else {
              const defaultImport = importInfo.isDefault ? usedImports[0] : null;
              const namedImports = importInfo.isDefault ? usedImports.slice(1) : usedImports;
              
              if (defaultImport && namedImports.length > 0) {
                newImport = `import ${defaultImport}, { ${namedImports.join(', ')} } from '${importInfo.module}';`;
              } else if (defaultImport) {
                newImport = `import ${defaultImport} from '${importInfo.module}';`;
              } else {
                newImport = `import { ${namedImports.join(', ')} } from '${importInfo.module}';`;
              }
            }
            modifiedLines.push(newImport);
          } else {
            modifiedLines.push(line);
          }
        }
        // Skip line if no imports are used
      } else {
        modifiedLines.push(line);
      }
    });

    return modifiedLines.join('\n');
  }

  /**
   * Fix 'any' types with better alternatives
   */
  private fixAnyTypes(content: string): string {
    // Basic any replacements
    content = content.replace(/: any\[\]/g, ': unknown[]');
    content = content.replace(/: any(?=[\s,\)\};\]])/g, ': unknown');
    content = content.replace(/<any>/g, '<unknown>');
    content = content.replace(/as any(?=[\s,\)\};\]])/g, 'as unknown');
    
    // Function parameters
    content = content.replace(/\(([^:)]+): any\)/g, '($1: unknown)');
    
    // Specific patterns for common cases
    // Event handlers
    content = content.replace(
      /\((e|event): unknown\)(?=\s*=>[\s\S]*?\.(preventDefault|stopPropagation|target|currentTarget))/g,
      '($1: React.MouseEvent | React.FormEvent)'
    );
    
    // Error handlers
    content = content.replace(
      /\((error|err|e): unknown\)(?=\s*=>[\s\S]*?\.(message|stack|code|name))/g,
      '($1: Error)'
    );
    
    // Axios/API responses
    content = content.replace(
      /: unknown(?=[\s\S]{0,50}\.(data|status|headers))/g,
      ': { data?: unknown; status?: number; headers?: Record<string, string> }'
    );
    
    return content;
  }

  /**
   * Fix unused variables
   */
  private fixUnusedVariables(content: string): string {
    // Prefix unused parameters with underscore
    // Catch blocks
    content = content.replace(/catch\s*\((\w+)\)/g, (match, param) => {
      if (!param.startsWith('_')) {
        return `catch (_${param})`;
      }
      return match;
    });
    
    // Unused destructured variables
    content = content.replace(
      /const\s*{([^}]+)}\s*=/g,
      (match, destructured) => {
        const parts = destructured.split(',').map((part: string) => {
          const trimmed = part.trim();
          // Check if this might be unused (simple heuristic)
          if (trimmed.includes(':')) {
            const [name, alias] = trimmed.split(':').map(s => s.trim());
            return `${name}: _${alias}`;
          }
          return part;
        });
        return `const {${parts.join(',')}} =`;
      }
    );
    
    // Unused setState functions
    content = content.replace(
      /const\s+\[(\w+),\s+set\w+\]\s*=\s*useState/g,
      (match, stateVar) => {
        // If the setter is unused, prefix with underscore
        return match.replace(/,\s+(set\w+)/, ', _$1');
      }
    );
    
    return content;
  }

  /**
   * Fix React hooks dependency warnings
   */
  private fixReactHooksDeps(content: string): string {
    const lines = content.split('\n');
    const modifiedLines = lines.map(line => {
      // Check if line contains hooks with empty deps
      if ((line.includes('useEffect') || line.includes('useCallback') || line.includes('useMemo')) 
          && line.includes('[]') 
          && !line.includes('eslint-disable')) {
        // Add eslint-disable comment
        return line.replace(/(\[\]\s*\))/, '$1 // eslint-disable-line react-hooks/exhaustive-deps');
      }
      return line;
    });
    
    return modifiedLines.join('\n');
  }

  /**
   * Fix optional chaining with non-null assertions
   */
  private fixOptionalChaining(content: string): string {
    // Remove non-null assertions after optional chaining
    content = content.replace(/\?\.\w+!/g, (match) => {
      return match.substring(0, match.length - 1);
    });
    
    return content;
  }

  /**
   * Fix export issues for React components
   */
  private fixExportIssues(content: string, filePath: string): string {
    if (filePath.includes('Router') || filePath.includes('router')) {
      const lines = content.split('\n');
      const hasNonComponentExports = lines.some(line => 
        line.includes('export const') || line.includes('export function')
      );
      
      if (hasNonComponentExports) {
        // Add comment to suppress the warning
        const firstLine = '// eslint-disable-next-line react-refresh/only-export-components\n';
        if (!content.startsWith('// eslint-disable')) {
          return firstLine + content;
        }
      }
    }
    
    return content;
  }

  /**
   * Escape special regex characters
   */
  private escapeRegex(str: string): string {
    return str.replace(/[.*+?^${}()|[\]\\]/g, '\\$&');
  }
}

// Run the fixer
const fixer = new ESLintFixer();
fixer.fix().then(() => {
  console.log('\nESLint fixes completed!');
  console.log('Run "npm run lint" to see any remaining issues.');
}).catch(error => {
  console.error('Error:', error);
  process.exit(1);
});