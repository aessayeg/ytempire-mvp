const fs = require('fs');
const path = require('path');

// Pattern to fix JSX expression semicolons
const fixes = [
  // Remove semicolons from JSX expressions
  {
    pattern: /\{([^}]*)\.\s*toFixed\((\d+)\);\s*\}/g,
    replacement: '{$1.toFixed($2)}'
  },
  {
    pattern: /\{([^}]*)\.\s*toLocaleString\(\);\s*\}/g,
    replacement: '{$1.toLocaleString()}'
  },
  {
    pattern: /\)\s*;\s*\}/g,
    replacement: ')}'
  },
  // Fix malformed ternary closing
  {
    pattern: /\)\s*;\s*\}/g,
    replacement: ')}'
  },
  // Fix extra semicolons in JSX
  {
    pattern: /\};\s*</g,
    replacement: '}<'
  },
  // Fix includes checks with semicolon
  {
    pattern: /\.includes\(([^)]+)\);\s*\}/g,
    replacement: '.includes($1)}'
  }
];

function fixFile(filePath) {
  try {
    let content = fs.readFileSync(filePath, 'utf8');
    let modified = false;
    
    fixes.forEach(fix => {
      const newContent = content.replace(fix.pattern, fix.replacement);
      if (newContent !== content) {
        content = newContent;
        modified = true;
      }
    });
    
    if (modified) {
      fs.writeFileSync(filePath, content, 'utf8');
      console.log(`Fixed: ${filePath}`);
      return true;
    }
    return false;
  } catch (error) {
    console.error(`Error processing ${filePath}:`, error.message);
    return false;
  }
}

function findFiles(dir, extension) {
  const files = [];
  
  try {
    const entries = fs.readdirSync(dir, { withFileTypes: true });
    
    for (const entry of entries) {
      const fullPath = path.join(dir, entry.name);
      
      if (entry.isDirectory() && !['node_modules', '.git', 'dist', 'build'].includes(entry.name)) {
        files.push(...findFiles(fullPath, extension));
      } else if (entry.isFile() && entry.name.endsWith(extension)) {
        files.push(fullPath);
      }
    }
  } catch (error) {
    console.error(`Error reading directory ${dir}:`, error.message);
  }
  
  return files;
}

console.log('Fixing JSX semicolon issues...');

const srcDir = path.join(__dirname, '..', 'frontend', 'src');
const tsxFiles = findFiles(srcDir, '.tsx');
const tsFiles = findFiles(srcDir, '.ts');
const allFiles = [...tsxFiles, ...tsFiles];

console.log(`Found ${allFiles.length} TypeScript files to check`);

let fixedCount = 0;
for (const file of allFiles) {
  if (fixFile(file)) {
    fixedCount++;
  }
}

console.log(`Fixed ${fixedCount} files`);