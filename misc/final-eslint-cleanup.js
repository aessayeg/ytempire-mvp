#!/usr/bin/env node
/**
 * Final ESLint cleanup - removes all unused imports
 */

const fs = require('fs');
const path = require('path');
const { execSync } = require('child_process');

const frontendSrc = path.join(__dirname, '..', 'frontend', 'src');

function removeAllUnusedImports(filePath) {
  try {
    let content = fs.readFileSync(filePath, 'utf8');
    const lines = content.split('\n');
    
    // Get all import statements with their line numbers
    const imports = [];
    lines.forEach((line, index) => {
      if (line.trim().startsWith('import ')) {
        imports.push({ line, index });
      }
    });
    
    // Get the code without imports
    const codeLines = lines.filter((line, index) => 
      !imports.some(imp => imp.index === index)
    ).join('\n');
    
    // Filter imports - keep only those that are used
    const filteredLines = [];
    lines.forEach((line, index) => {
      const importData = imports.find(imp => imp.index === index);
      
      if (importData) {
        // Extract imported items
        let shouldKeep = false;
        
        // Check for default imports
        const defaultMatch = line.match(/import\s+(\w+)\s+from/);
        if (defaultMatch) {
          const importName = defaultMatch[1];
          const regex = new RegExp(`\\b${importName}\\b`);
          if (regex.test(codeLines)) {
            shouldKeep = true;
          }
        }
        
        // Check for named imports
        const namedMatch = line.match(/import\s+{([^}]+)}/);
        if (namedMatch) {
          const imports = namedMatch[1].split(',').map(s => s.trim());
          const usedImports = imports.filter(imp => {
            const cleanName = imp.split(' as ')[0].trim();
            const regex = new RegExp(`\\b${cleanName}\\b`);
            return regex.test(codeLines);
          });
          
          if (usedImports.length > 0) {
            if (usedImports.length === imports.length) {
              shouldKeep = true;
            } else {
              // Reconstruct with only used imports
              const moduleMatch = line.match(/from\s+['"]([^'"]+)['"]/);
              if (moduleMatch) {
                const newImport = `import { ${usedImports.join(', ')} } from '${moduleMatch[1]}';`;
                filteredLines.push(newImport);
                return;
              }
            }
          }
        }
        
        // Check for namespace imports
        const namespaceMatch = line.match(/import\s+\*\s+as\s+(\w+)/);
        if (namespaceMatch) {
          const importName = namespaceMatch[1];
          const regex = new RegExp(`\\b${importName}\\b`);
          if (regex.test(codeLines)) {
            shouldKeep = true;
          }
        }
        
        // Check for side-effect imports
        if (line.match(/import\s+['"][^'"]+['"]/)) {
          shouldKeep = true; // Keep side-effect imports
        }
        
        if (shouldKeep) {
          filteredLines.push(line);
        }
      } else {
        filteredLines.push(line);
      }
    });
    
    const newContent = filteredLines.join('\n');
    if (newContent !== content) {
      fs.writeFileSync(filePath, newContent, 'utf8');
      return true;
    }
    return false;
  } catch (error) {
    console.error(`Error processing ${filePath}:`, error.message);
    return false;
  }
}

function getAllFiles(dir) {
  const files = [];
  
  function walk(currentDir) {
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
  }
  
  walk(dir);
  return files;
}

console.log('Starting final ESLint cleanup...');
console.log('-'.repeat(50));

const files = getAllFiles(frontendSrc);
let totalFixed = 0;

files.forEach(file => {
  if (removeAllUnusedImports(file)) {
    const relPath = path.relative(frontendSrc, file);
    console.log(`Fixed: ${relPath}`);
    totalFixed++;
  }
});

console.log('-'.repeat(50));
console.log(`Total files fixed: ${totalFixed}`);

// Run ESLint to show final state
console.log('\nRunning ESLint to show final state...');
try {
  execSync('cd frontend && npm run lint', { stdio: 'inherit' });
} catch (error) {
  // ESLint will exit with error if issues remain
}

console.log('\nCleanup completed!');