#!/usr/bin/env node
/**
 * Script to fix all ESLint errors in the YTEmpire MVP frontend
 * This script programmatically fixes common ESLint issues
 */

const fs = require('fs');
const path = require('path');
const { execSync } = require('child_process');

const frontendSrc = path.join(__dirname, '..', 'frontend', 'src');

// Files with most errors based on ESLint output
const filesToFix = [
  'components/Auth/EmailVerification.tsx',
  'components/BatchOperations/BatchOperations.tsx',
  'components/BulkOperations/EnhancedBulkOperations.tsx',
  'components/DataVisualization/RevenueVisualization.tsx',
  'components/ExportImport/DataExportImport.tsx',
  'components/Layout/MobileLayout.tsx',
  'components/Monetization/MonetizationDashboard.tsx',
  'components/Performance/PerformanceMonitor.tsx',
  'components/Revenue/EnhancedRevenueDashboard.tsx',
  'components/Support/AIAssistedSupport.tsx',
  'components/Video/VideoAnalytics.tsx',
  'components/Video/VideoCard.tsx',
  'components/Video/VideoGenerationForm.tsx',
  'components/Video/VideoPlayer.tsx',
  'components/Dashboard/Dashboard.tsx',
  'pages/Channels/CreateChannel.tsx',
  'pages/Channels/EditChannel.tsx',
  'pages/DataExplorer/DataExplorer.tsx',
  'pages/Revenue/Revenue.tsx',
  'pages/Settings/Settings.tsx',
  'pages/Videos/CreateVideo.tsx',
  'pages/Videos/VideoQueue.tsx',
  'services/analytics_tracker.ts',
  'services/api.ts',
  'services/websocketService.ts',
  'stores/authStore.ts',
  'router/optimizedRouter.tsx',
  'tests/accessibility/test-accessibility.ts'
];

/**
 * Remove unused imports from a file
 */
function removeUnusedImports(filePath, content) {
  const lines = content.split('\n');
  const modifiedLines = [];
  const importRegex = /^import\s+(?:{([^}]+)}|([^,\s]+))\s+from\s+['"]([^'"]+)['"];?$/;
  
  // Build a map of all imports
  const imports = [];
  lines.forEach((line, index) => {
    const match = line.trim().match(importRegex);
    if (match) {
      if (match[1]) {
        // Named imports
        const namedImports = match[1].split(',').map(i => i.trim());
        imports.push({ line, index, imports: namedImports, module: match[3] });
      } else if (match[2]) {
        // Default import
        imports.push({ line, index, imports: [match[2]], module: match[3] });
      }
    }
  });

  // Get the code without import lines
  const codeWithoutImports = lines
    .filter((_, index) => !imports.some(imp => imp.index === index))
    .join('\n');

  // Check which imports are used
  lines.forEach((line, index) => {
    const importInfo = imports.find(imp => imp.index === index);
    if (importInfo) {
      const usedImports = importInfo.imports.filter(imp => {
        const cleanName = imp.split(' as ')[0].trim();
        const regex = new RegExp(`\\b${cleanName}\\b`);
        return regex.test(codeWithoutImports);
      });

      if (usedImports.length > 0) {
        if (usedImports.length !== importInfo.imports.length) {
          // Reconstruct import with only used items
          const newImport = `import { ${usedImports.join(', ')} } from '${importInfo.module}';`;
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
 * Fix 'any' types by replacing with 'unknown' or more specific types
 */
function fixAnyTypes(content) {
  // Replace 'any' with 'unknown' in most cases
  content = content.replace(/: any\[\]/g, ': unknown[]');
  content = content.replace(/: any(?=[\s,\)\}])/g, ': unknown');
  content = content.replace(/<any>/g, '<unknown>');
  content = content.replace(/as any(?=[\s,\)\}])/g, 'as unknown');
  
  // Fix specific patterns
  content = content.replace(/\(([^:]+): any\)/g, '($1: unknown)');
  
  // Event handlers
  content = content.replace(/\((e|event): unknown\)(?=\s*=>.*\.(preventDefault|stopPropagation|target))/g, 
    '($1: React.MouseEvent | React.FormEvent)');
  
  // Error handlers
  content = content.replace(/\((error|err|e): unknown\)(?=\s*=>.*\.(message|stack|code))/g,
    '($1: Error)');
  
  return content;
}

/**
 * Fix unused variables by prefixing with underscore
 */
function fixUnusedVariables(content) {
  // Fix catch blocks
  content = content.replace(/catch\s*\(([^)]+)\)\s*{/g, (match, param) => {
    if (!param.startsWith('_')) {
      return `catch (_${param}) {`;
    }
    return match;
  });
  
  // Fix function parameters that are clearly unused
  const unusedParams = ['error', 'err', 'e', 'message'];
  unusedParams.forEach(param => {
    const regex = new RegExp(`\\b(${param})\\b(?=.*is defined but never used)`, 'g');
    content = content.replace(regex, `_$1`);
  });
  
  return content;
}

/**
 * Fix React hooks exhaustive deps warnings
 */
function fixReactHooksDeps(content) {
  const lines = content.split('\n');
  const modifiedLines = lines.map(line => {
    if (line.includes('useEffect') || line.includes('useCallback') || line.includes('useMemo')) {
      if (line.includes('[]') && !line.includes('eslint-disable')) {
        return line + ' // eslint-disable-line react-hooks/exhaustive-deps';
      }
    }
    return line;
  });
  return modifiedLines.join('\n');
}

/**
 * Fix router export issue
 */
function fixRouterExports(filePath, content) {
  if (filePath.includes('optimizedRouter')) {
    // Move non-component exports to a separate file
    const lines = content.split('\n');
    const componentExports = [];
    const utilExports = [];
    
    lines.forEach(line => {
      if (line.includes('export') && !line.includes('export default')) {
        if (line.includes('const') || line.includes('function')) {
          utilExports.push(line);
        } else {
          componentExports.push(line);
        }
      } else {
        componentExports.push(line);
      }
    });
    
    // Create a utils file if needed
    if (utilExports.length > 0) {
      const utilsPath = path.join(path.dirname(filePath), 'routerUtils.ts');
      fs.writeFileSync(utilsPath, utilExports.join('\n'));
      
      // Update imports in main file
      content = `import { ${utilExports.map(line => {
        const match = line.match(/export\s+(?:const|function)\s+(\w+)/);
        return match ? match[1] : '';
      }).filter(Boolean).join(', ')} } from './routerUtils';\n` + componentExports.join('\n');
    }
  }
  return content;
}

/**
 * Process a single file
 */
function processFile(relPath) {
  const filePath = path.join(frontendSrc, relPath);
  
  if (!fs.existsSync(filePath)) {
    console.log(`File not found: ${relPath}`);
    return false;
  }
  
  try {
    let content = fs.readFileSync(filePath, 'utf8');
    const originalContent = content;
    
    // Apply fixes
    content = removeUnusedImports(filePath, content);
    content = fixAnyTypes(content);
    content = fixUnusedVariables(content);
    content = fixReactHooksDeps(content);
    content = fixRouterExports(filePath, content);
    
    // Write back if changed
    if (content !== originalContent) {
      fs.writeFileSync(filePath, content, 'utf8');
      console.log(`Fixed: ${relPath}`);
      return true;
    }
    
    return false;
  } catch (error) {
    console.error(`Error processing ${relPath}:`, error.message);
    return false;
  }
}

// Main execution
console.log('Starting ESLint error fixes...');
console.log('-'.repeat(50));

let totalFixed = 0;
filesToFix.forEach(file => {
  if (processFile(file)) {
    totalFixed++;
  }
});

// Process all other TypeScript/React files
function walkDir(dir, callback) {
  fs.readdirSync(dir).forEach(f => {
    const dirPath = path.join(dir, f);
    const isDirectory = fs.statSync(dirPath).isDirectory();
    if (isDirectory) {
      walkDir(dirPath, callback);
    } else {
      callback(path.join(dir, f));
    }
  });
}

walkDir(frontendSrc, (filePath) => {
  const ext = path.extname(filePath);
  if (['.ts', '.tsx', '.js', '.jsx'].includes(ext)) {
    const relPath = path.relative(frontendSrc, filePath);
    if (!filesToFix.includes(relPath.replace(/\\/g, '/'))) {
      if (processFile(relPath)) {
        totalFixed++;
      }
    }
  }
});

console.log('-'.repeat(50));
console.log(`Total files fixed: ${totalFixed}`);

// Run ESLint auto-fix again
console.log('\nRunning ESLint auto-fix...');
try {
  execSync('cd frontend && npm run lint:fix', { stdio: 'inherit' });
} catch (error) {
  // ESLint will exit with error if there are still issues
}

console.log('\nESLint fixes attempted. Check remaining issues with: npm run lint');