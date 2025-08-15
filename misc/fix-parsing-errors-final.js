#!/usr/bin/env node
/**
 * Fix Final Parsing Errors
 * Target remaining parsing errors in utility files
 */

const fs = require('fs');
const path = require('path');

const srcDir = path.join(__dirname, '..', 'frontend', 'src');

// Specific parsing error fixes
const parsingFixes = [
  // Fix declaration or statement expected errors
  {
    pattern: /export \{[^}]*\}\s*;\s*$/gm,
    replacement: (match) => match.trim()
  },

  // Fix comma expected errors in function parameters
  {
    pattern: /\([^)]*[^,]\s+[^)]*\)/g,
    replacement: (match) => {
      // Add comma between parameters if missing
      return match.replace(/(\w+)\s+(\w+\s*[):,])/g, '$1, $2');
    }
  },

  // Fix malformed object destructuring
  {
    pattern: /\{\s*([^}]+),\s*\}/g,
    replacement: (match, content) => {
      // Remove trailing comma before closing brace
      return `{ ${content.replace(/,\s*$/, '')} }`;
    }
  },

  // Fix malformed array destructuring
  {
    pattern: /\[\s*([^\]]+),\s*\]/g,
    replacement: (match, content) => {
      // Remove trailing comma before closing bracket
      return `[ ${content.replace(/,\s*$/, '')} ]`;
    }
  },

  // Fix incomplete arrow functions
  {
    pattern: /=>\s*$/gm,
    replacement: '=> {}'
  },

  // Fix incomplete function calls
  {
    pattern: /\(\s*,/g,
    replacement: '('
  },

  // Fix incomplete template literals
  {
    pattern: /\$\{[^}]*$/gm,
    replacement: (match) => match + '}'
  },

  // Fix incomplete object literals
  {
    pattern: /:\s*,/g,
    replacement: ': undefined,'
  }
];

// File-specific fixes
const specificFileFixes = {
  'EventEmitter.ts': [
    // Replace '_error' with '_'
    {
      pattern: /_error/g,
      replacement: '_'
    }
  ]
};

function processFile(filePath) {
  try {
    let content = fs.readFileSync(filePath, 'utf8');
    const fileName = path.basename(filePath);
    let hasChanges = false;

    // Apply general parsing fixes
    parsingFixes.forEach(fix => {
      const oldContent = content;
      if (typeof fix.replacement === 'function') {
        content = content.replace(fix.pattern, fix.replacement);
      } else {
        content = content.replace(fix.pattern, fix.replacement);
      }
      if (content !== oldContent) {
        hasChanges = true;
        console.log(`✓ Applied parsing fix in ${fileName}`);
      }
    });

    // Apply file-specific fixes
    if (specificFileFixes[fileName]) {
      specificFileFixes[fileName].forEach(fix => {
        const oldContent = content;
        content = content.replace(fix.pattern, fix.replacement);
        if (content !== oldContent) {
          hasChanges = true;
          console.log(`✓ Applied specific fix in ${fileName}: ${fix.pattern.toString()}`);
        }
      });
    }

    // Special handling for common parsing issues
    
    // Fix incomplete exports
    content = content.replace(/export\s*;/g, '');
    
    // Fix incomplete imports
    content = content.replace(/import\s*;/g, '');
    
    // Fix incomplete interface/type definitions
    content = content.replace(/interface\s+\w+\s*$/gm, (match) => match + ' {}');
    content = content.replace(/type\s+\w+\s*=\s*$/gm, (match) => match + 'unknown;');

    // Fix incomplete function definitions
    content = content.replace(/function\s+\w+\s*\([^)]*\)\s*$/gm, (match) => match + ' {}');

    // Fix stray semicolons causing parsing errors
    content = content.replace(/;\s*;/g, ';');
    content = content.replace(/^\s*;\s*$/gm, '');

    if (hasChanges || content !== fs.readFileSync(filePath, 'utf8')) {
      fs.writeFileSync(filePath, content);
      console.log(`✅ Fixed parsing errors in ${fileName}`);
    }

  } catch (error) {
    console.error(`❌ Error processing ${filePath}:`, error.message);
  }
}

function walkDir(dir) {
  const files = fs.readdirSync(dir);
  
  files.forEach(file => {
    const fullPath = path.join(dir, file);
    const stat = fs.statSync(fullPath);
    
    if (stat.isDirectory()) {
      walkDir(fullPath);
    } else if (file.match(/\.(tsx?|jsx?)$/)) {
      processFile(fullPath);
    }
  });
}

console.log('🔧 Starting final parsing error fixes...');
walkDir(srcDir);
console.log('✅ Final parsing error fixes completed!');