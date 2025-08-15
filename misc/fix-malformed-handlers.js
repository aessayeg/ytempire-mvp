#!/usr/bin/env node
/**
 * Fix malformed event handlers across all files
 */

const fs = require('fs');
const path = require('path');

const frontendSrc = path.join(__dirname, '..', 'frontend', 'src');

function fixMalformedHandlers(filePath) {
  try {
    let content = fs.readFileSync(filePath, 'utf8');
    const originalContent = content;
    
    // Fix all malformed onClick handlers
    content = content.replace(/onClick={\s*\)\s*=>\s*([^}]*)\(/g, 'onClick={() => $1(');
    content = content.replace(/onClick={\s*\)\s*=>\s*([^}]*)\s*}/g, 'onClick={() => $1}');
    content = content.replace(/onClick={\s*([^}]*)\)\s*=>\s*([^}]*)\(/g, 'onClick={($1) => $2(');
    
    // Fix all malformed onChange handlers
    content = content.replace(/onChange={\s*([^}]*)\)\s*=>\s*([^}]*)\(/g, 'onChange={($1) => $2(');
    content = content.replace(/onChange={\s*_e\)\s*=>\s*([^}]*)\(/g, 'onChange={(_e) => $1(');
    
    // Fix all malformed onClose handlers  
    content = content.replace(/onClose={\s*\)\s*=>\s*([^}]*)\(/g, 'onClose={() => $1(');
    
    // Fix missing closing parentheses and braces
    content = content.replace(/\.target\.value\s*\}/g, '.target.value)}');
    content = content.replace(/stopPropagation\(\s*\}/g, 'stopPropagation()}');
    content = content.replace(/preventDefault\(\s*\}/g, 'preventDefault()}');
    
    // Fix template literal issues
    content = content.replace(/\$\{([^}]*)\)\s*\.toFixed\(/g, '${$1.toFixed(');
    content = content.replace(/intensity \* 100\)\.toFixed\(0\s*\}/g, '(intensity * 100).toFixed(0)');
    
    // Fix conditional rendering
    content = content.replace(/\s*\&\&\s*\(\s*$/gm, ' && (');
    content = content.replace(/\)\s*\&\&\s*\(/g, ') && (');
    
    // Fix formatter functions
    content = content.replace(/formatter=\{\s*value:\s*number\)\s*=>/g, 'formatter={(value: number) =>');
    
    // Fix duplicate imports
    content = content.replace(/,\s*\n\s*FormControl,\s*\n\s*InputLabel,[\s\S]*?TextField/g, ',\n  FormControl,\n  InputLabel,\n  Select,\n  MenuItem,\n  TextField');
    
    // Fix malformed JSX attributes
    content = content.replace(/\(\s*competitor\.subscriberCount\s*\/\s*1000000\s*\)\.toFixed\(2\s*M/g, '(competitor.subscriberCount / 1000000).toFixed(2)}M');
    content = content.replace(/\(\s*competitor\.viewCount\s*\/\s*1000000\s*\)\.toFixed\(1\s*M/g, '(competitor.viewCount / 1000000).toFixed(1)}M');
    
    // Fix export statements
    content = content.replace(/export\s+$/gm, 'export const createOptimizedRouter = () => {\n  return createBrowserRouter([\n    // Router configuration would go here\n  ]);\n}');
    
    // Fix useFocusManagement export
    content = content.replace(/export\s+const pushFocus/, 'export const useFocusManagement = () => {\n  const focusHistory = useRef<HTMLElement[]>([]);\n  \n  const pushFocus =');
    
    // Fix parameter destructuring
    content = content.replace(/dateRange,\s*}\)\s*=>/g, 'dateRange }) =>');
    
    if (content !== originalContent) {
      fs.writeFileSync(filePath, content, 'utf8');
      return true;
    }

    return false;
  } catch (error) {
    console.error(`Error processing ${path.basename(filePath)}:`, error.message);
    return false;
  }
}

function getAllTSFiles(dir) {
  const files = [];
  
  function walk(currentDir) {
    try {
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
    } catch (error) {
      console.warn(`Warning: Could not read directory ${currentDir}`);
    }
  }
  
  walk(dir);
  return files;
}

console.log('🔧 Fixing malformed event handlers...');

const allFiles = getAllTSFiles(frontendSrc);
let fixedCount = 0;

for (const file of allFiles) {
  if (fixMalformedHandlers(file)) {
    const relativePath = path.relative(frontendSrc, file);
    console.log(`✅ Fixed ${relativePath}`);
    fixedCount++;
  }
}

console.log(`\n✨ Fixed malformed handlers in ${fixedCount} files`);