#!/usr/bin/env node
/**
 * Fix parsing errors caused by broken import statements
 */

const fs = require('fs');
const path = require('path');
const { execSync } = require('child_process');

const frontendSrc = path.join(__dirname, '..', 'frontend', 'src');

function fixBrokenImports(filePath) {
  try {
    let content = fs.readFileSync(filePath, 'utf8');
    const lines = content.split('\n');
    const fixedLines = [];
    
    let i = 0;
    while (i < lines.length) {
      const line = lines[i];
      
      // Check if this line looks like a broken import (starts with spaces and has import-like content)
      if (i > 0 && 
          line.trim() && 
          !line.trim().startsWith('import') && 
          !line.trim().startsWith('//') && 
          !line.trim().startsWith('/*') &&
          !line.trim().startsWith('*') &&
          !line.trim().startsWith('export') &&
          !line.trim().startsWith('const') &&
          !line.trim().startsWith('let') &&
          !line.trim().startsWith('var') &&
          !line.trim().startsWith('function') &&
          !line.trim().startsWith('class') &&
          !line.trim().startsWith('interface') &&
          !line.trim().startsWith('type') &&
          !line.trim().startsWith('enum') &&
          !line.includes('=') &&
          !line.includes('{') &&
          !line.includes('}') &&
          !line.includes('(') &&
          !line.includes(')') &&
          !line.includes('<') &&
          !line.includes('>') &&
          (line.includes(',') || line.trim().endsWith(';'))) {
        
        // Look backwards to find the start of a broken import
        let j = i - 1;
        let importStart = -1;
        while (j >= 0) {
          const prevLine = lines[j];
          if (prevLine.trim().startsWith('import') && 
              (prevLine.includes('{') || prevLine.includes('from'))) {
            importStart = j;
            break;
          } else if (prevLine.trim() === '' || 
                     prevLine.trim().endsWith(',') ||
                     (!prevLine.trim().startsWith('import') && 
                      !prevLine.includes('from') && 
                      (prevLine.includes(',') || prevLine.trim().match(/^[A-Z][a-zA-Z0-9]*,?$/)))) {
            j--;
          } else {
            break;
          }
        }
        
        if (importStart >= 0) {
          // Collect all lines that are part of this broken import
          let importLines = [];
          let k = importStart;
          while (k <= i && k < lines.length) {
            const currentLine = lines[k];
            if (currentLine.trim()) {
              importLines.push(currentLine.trim());
            }
            k++;
          }
          
          // Try to reconstruct the import
          if (importLines.length > 0) {
            let reconstructed = importLines.join(' ');
            
            // Clean up the import statement
            reconstructed = reconstructed.replace(/\s+/g, ' ');
            reconstructed = reconstructed.replace(/,\s*}/g, ' }');
            reconstructed = reconstructed.replace(/{\s*,/g, '{ ');
            reconstructed = reconstructed.replace(/,\s*,/g, ',');
            
            // Skip lines that were part of the broken import
            for (let skip = importStart; skip <= i; skip++) {
              if (skip === importStart) {
                fixedLines.push(reconstructed);
              }
              // Skip other lines in the import block
            }
            
            i++; // Move to next line
            continue;
          }
        }
      }
      
      // Check for lines that are just hanging commas or identifiers
      if (line.trim() === ',' || 
          (line.trim().match(/^[A-Za-z][A-Za-z0-9]*,?$/) && 
           !line.includes('=') && 
           !line.includes('(') && 
           !line.includes('{') &&
           i > 0 && 
           !lines[i-1].trim().startsWith('const') &&
           !lines[i-1].trim().startsWith('let') &&
           !lines[i-1].trim().startsWith('var'))) {
        // Skip orphaned import fragments
        i++;
        continue;
      }
      
      // Check for lines that start with } from
      if (line.trim().startsWith('} from')) {
        // This is likely the end of a broken import, skip it
        i++;
        continue;
      }
      
      fixedLines.push(line);
      i++;
    }
    
    const newContent = fixedLines.join('\n');
    
    if (newContent !== content) {
      fs.writeFileSync(filePath, newContent, 'utf8');
      return true;
    }
    return false;
  } catch (error) {
    console.error(`Error fixing ${filePath}:`, error.message);
    return false;
  }
}

function getAllFiles(dir) {
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

console.log('Fixing parsing errors...');
console.log('-'.repeat(50));

const files = getAllFiles(frontendSrc);
let totalFixed = 0;

files.forEach(file => {
  if (fixBrokenImports(file)) {
    const relPath = path.relative(frontendSrc, file);
    console.log(`Fixed: ${relPath}`);
    totalFixed++;
  }
});

console.log('-'.repeat(50));
console.log(`Total files fixed: ${totalFixed}`);

console.log('\nRunning ESLint to check progress...');
try {
  const result = execSync('cd frontend && npm run lint', { encoding: 'utf8' });
  console.log(result);
} catch (error) {
  // Show the last few lines with error summary
  const output = error.stdout || error.message;
  const lines = output.split('\n');
  console.log(lines.slice(-10).join('\n'));
}