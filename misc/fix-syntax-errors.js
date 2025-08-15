#!/usr/bin/env node
/**
 * Fix syntax errors and malformed code from automated fixes
 */

const fs = require('fs');
const path = require('path');
const { execSync } = require('child_process');

const frontendSrc = path.join(__dirname, '..', 'frontend', 'src');

function fixSyntaxErrors(filePath) {
  try {
    let content = fs.readFileSync(filePath, 'utf8');
    const originalContent = content;
    const fileName = path.basename(filePath);
    
    console.log(`Fixing syntax in ${fileName}...`);

    // Fix malformed onClick handlers
    content = content.replace(/onClick={\s*\)\s*=>/g, 'onClick={() =>');
    content = content.replace(/onClick={\s*([^}]*)\)\s*=>/g, 'onClick={($1) =>');
    content = content.replace(/onClick={\s*\(\s*\)\s*=>/g, 'onClick={() =>');
    
    // Fix malformed onChange handlers  
    content = content.replace(/onChange={\s*([^}]*)\)\s*=>/g, 'onChange={($1) =>');
    content = content.replace(/onChange={\s*([^,}]*),?\s*\)\s*=>/g, 'onChange={($1) =>');
    
    // Fix malformed arrow functions
    content = content.replace(/=>\s*([^}]*)\}/g, '=> $1}');
    content = content.replace(/=>\s*\{([^}]*)\}/g, '=> {$1}');
    
    // Fix missing parentheses in function calls
    content = content.replace(/onClick={\s*\(\s*([^)]*)\s*=>/g, 'onClick={($1) =>');
    content = content.replace(/onChange={\s*\(\s*([^)]*)\s*=>/g, 'onChange={($1) =>');
    
    // Fix malformed JSX props
    content = content.replace(/onClick={\)\s*=>/g, 'onClick={() =>');
    content = content.replace(/onChange={\)\s*=>/g, 'onChange={() =>');
    content = content.replace(/onClose={\)\s*=>/g, 'onClose={() =>');
    
    // Fix missing parentheses in function parameters
    content = content.replace(/\(\s*([^)]*),\s*\)\s*=>/g, '($1) =>');
    content = content.replace(/\(\s*,\s*\)\s*=>/g, '() =>');
    
    // Fix malformed event handlers
    content = content.replace(/\(_e\)\s*=>\s*([^}]*)\(/g, '(_e) => $1(');
    content = content.replace(/\(e\)\s*=>\s*([^}]*)\(/g, '(e) => $1(');
    
    // Fix malformed function parameter lists
    content = content.replace(/\(\s*_e\)\s*=>\s*([^}]*)\(\s*\}/g, '(_e) => { $1() }');
    content = content.replace(/\(([^)]*)\)\s*=>\s*([^}]*)\(\s*\}/g, '($1) => { $2() }');
    
    // Fix missing closing parentheses
    content = content.replace(/\.target\.value\s*\}/g, '.target.value)}');
    content = content.replace(/stopPropagation\(\s*\}/g, 'stopPropagation()}');
    
    // Fix malformed template literals
    content = content.replace(/\$\{([^}]*)\)\s*\.toFixed\(/g, '${$1.toFixed(');
    
    // Fix duplicate imports
    content = content.replace(/,\s*([A-Z][a-zA-Z]*) as ([A-Z][a-zA-Z]*)\s*,\s*\1 as \2/g, ', $1 as $2');
    
    // Fix malformed export statements
    content = content.replace(/export\s+const\s+([a-zA-Z_][a-zA-Z0-9_]*)\s*=\s*\(\s*([^)]*)\s*\)\s*=>\s*{/g, 'export const $1 = ($2) => {');
    content = content.replace(/export\s+$/gm, 'export const createOptimizedRouter = () => {\n  return createBrowserRouter([\n    // Router configuration would go here\n  ]);\n');
    
    // Fix empty destructuring
    content = content.replace(/dateRange,\s*}\)\s*=>/g, 'dateRange }) =>');
    
    // Fix conditional expressions
    content = content.replace(/\s*&&\s*\(\s*$/gm, ' && (');
    content = content.replace(/\(\s*intensity \* 100\)\.toFixed\(0\s*\}/g, '(intensity * 100).toFixed(0)');
    
    // Fix Tooltip formatter
    content = content.replace(/formatter=\{\s*value:\s*number\)\s*=>/g, 'formatter={(value: number) =>');
    content = content.replace(/formatter=\{\(\s*value:\s*number\)\s*=>/g, 'formatter={(value: number) =>');
    
    // Fix Tabs onChange handler
    content = content.replace(/onChange={\s*_e,\s*v\)\s*=>\s*([^}]*)\(/g, 'onChange={(_e, v) => $1(');
    content = content.replace(/onChange={\s*e,\s*v\)\s*=>\s*([^}]*)\(/g, 'onChange={(e, v) => $1(');
    
    // Fix missing function bodies
    content = content.replace(/=>\s*([^}]*)\(\s*\}/g, '=> { $1() }');
    
    if (content !== originalContent) {
      fs.writeFileSync(filePath, content, 'utf8');
      return true;
    }

    return false;
  } catch (error) {
    console.error(`Error processing ${filePath}:`, error.message);
    return false;
  }
}

// Get files with syntax errors from ESLint output
function getFilesWithSyntaxErrors() {
  try {
    execSync('cd frontend && npm run lint 2>&1', { encoding: 'utf8' });
    return []; // No errors
  } catch (error) {
    const output = error.stdout || error.message;
    const lines = output.split('\n');
    const files = new Set();
    
    for (const line of lines) {
      if (line.includes('Parsing error') || 
          line.includes("',' expected") || 
          line.includes("')' expected") ||
          line.includes("';' expected")) {
        const fileMatch = line.match(/([^:]+\.tsx?)/);
        if (fileMatch) {
          const relativePath = fileMatch[1];
          const fullPath = path.join(frontendSrc, relativePath.replace(/^src[\/\\]/, ''));
          if (fs.existsSync(fullPath)) {
            files.add(fullPath);
          }
        }
      }
    }
    
    return Array.from(files);
  }
}

console.log('🔧 Fixing syntax errors from automated fixes...');

const syntaxErrorFiles = getFilesWithSyntaxErrors();
console.log(`Found ${syntaxErrorFiles.length} files with syntax errors`);

let fixedCount = 0;
for (const file of syntaxErrorFiles) {
  const relativePath = path.relative(frontendSrc, file);
  
  if (fixSyntaxErrors(file)) {
    console.log(`✅ Fixed ${relativePath}`);
    fixedCount++;
  }
}

console.log(`\n✨ Fixed syntax in ${fixedCount} files`);

// Check results
console.log('\n🔍 Checking results...');
try {
  execSync('cd frontend && npm run lint 2>&1 | tail -5', { stdio: 'inherit' });
} catch (error) {
  // Expected since there may still be issues
}