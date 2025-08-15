const fs = require('fs');
const path = require('path');

// Comprehensive fix patterns for remaining ESLint errors
const fixes = [
  // Fix missing closing braces and semicolons in object returns
  {
    pattern: /(\s+rpm: [^}]+)\s*}\);/g,
    replacement: '$1\n      };\n    });'
  },
  
  // Fix template literal issues with backticks in wrong places  
  {
    pattern: /id: `([^`]+)`,`/g,
    replacement: 'id: `$1`,'
  },
  
  // Fix template literal issues with titles
  {
    pattern: /title: `([^`]+)`,`/g,
    replacement: 'title: `$1`,'
  },
  
  // Fix malformed CSS values like "8884 d8"
  {
    pattern: /'#8884 d8'/g,
    replacement: "'#8884d8'"
  },
  {
    pattern: /'#82 ca9 d'/g,
    replacement: "'#82ca9d'"
  },
  {
    pattern: /'#ff7 c7 c'/g,
    replacement: "'#ff7c7c'"
  },
  
  // Fix unterminated template literals
  {
    pattern: /labelFormatter=\{\(date\) => format\(parseISO\(date\), 'PPP'\)`/g,
    replacement: "labelFormatter={(date) => format(parseISO(date), 'PPP')}"
  },
  
  // Fix malformed Pie chart props
  {
    pattern: /labelLine=\{false\}`/g,
    replacement: 'labelLine={false}'
  },
  
  // Fix incomplete color definitions
  {
    pattern: /fill="#8884 d8"/g,
    replacement: 'fill="#8884d8"'
  },
  
  // Fix malformed array maps
  {
    pattern: /\.map\(\(([^)]+)\) => \(`/g,
    replacement: '.map(($1) => ('
  },
  
  // Fix speed dial handlers
  {
    pattern: /onOpen=\{\(\) => setSpeedDialOpen\(true}/g,
    replacement: 'onOpen={() => setSpeedDialOpen(true)}'
  },
  {
    pattern: /onClose=\{\(\) => setSpeedDialOpen\(false}/g,
    replacement: 'onClose={() => setSpeedDialOpen(false)}'
  },
  
  // Fix missing closing parentheses in function calls
  {
    pattern: /onClick=\{\(\(\) => ([^}]+)}/g,
    replacement: 'onClick={() => $1}'
  },
  
  // Fix spacing issues in numbers
  {
    pattern: /(\d+)\s+([a-zA-Z]+)/g,
    replacement: function(match, num, unit) {
      if (['px', 's', 'ms', '%', 'd8', 'ca9', 'c7c'].includes(unit.toLowerCase())) {
        return num + unit;
      }
      return match;
    }
  },
  
  // Fix Chip props with backticks
  {
    pattern: /<Chip`/g,
    replacement: '<Chip'
  },
  
  // Fix unused variable warnings by replacing with underscore
  {
    pattern: /catch \(([a-zA-Z_][a-zA-Z0-9_]*)\) \{/g,
    replacement: 'catch (_) {'
  },
  
  // Fix incomplete object declarations
  {
    pattern: /\{\s*([a-zA-Z_][a-zA-Z0-9_]*)\s*}/g,
    replacement: function(match, variable) {
      if (match.includes('{') && match.includes('}') && !match.includes(':')) {
        return match; // Keep as is if it looks like destructuring
      }
      return match;
    }
  },
  
  // Fix malformed function parameters
  {
    pattern: /onChange=\{\(\(([^)]+)\) => ([^}]+)}/g,
    replacement: 'onChange={($1) => $2}'
  },
  
  // Fix template literal syntax errors
  {
    pattern: /secondary=\{`\$\{([^}]+)\.toFixed\(2\)`\}/g,
    replacement: 'secondary={`$${$1.toFixed(2)}`}'
  },
  
  // Fix missing closing braces in array operations
  {
    pattern: /\.filter\(([^)]+)\) => \{([^}]+)\s*}\s*\)/g,
    replacement: '.filter($1 => $2)'
  },
  
  // Fix incomplete color array syntax
  {
    pattern: /colors\[\s*index\s*%\s*colors\.length\s*\]/g,
    replacement: 'colors[index % colors.length]'
  },
  
  // Fix missing imports
  {
    pattern: /import\s*{\s*([^}]*)\s*}\s*from\s*'@mui\/material';/g,
    replacement: function(match, imports) {
      // Remove duplicates and clean up
      const importList = imports.split(',').map(imp => imp.trim()).filter(Boolean);
      const uniqueImports = [...new Set(importList)];
      return `import {\n  ${uniqueImports.join(',\n  ')}\n} from '@mui/material';`;
    }
  }
];

// Apply targeted fixes to specific files
const specificFixes = [
  {
    file: 'AnalyticsDashboard.tsx',
    fixes: [
      {
        pattern: /rpm: 3 \+ Math\.random\(\) \* 2\s*}\);/,
        replacement: 'rpm: 3 + Math.random() * 2\n      };\n    });'
      }
    ]
  }
];

function applyFixes(content, filename) {
  let fixedContent = content;
  
  // Apply general fixes
  fixes.forEach(fix => {
    if (typeof fix.replacement === 'function') {
      fixedContent = fixedContent.replace(fix.pattern, fix.replacement);
    } else {
      fixedContent = fixedContent.replace(fix.pattern, fix.replacement);
    }
  });
  
  // Apply file-specific fixes
  const baseName = path.basename(filename);
  const fileSpecific = specificFixes.find(sf => sf.file === baseName);
  if (fileSpecific) {
    fileSpecific.fixes.forEach(fix => {
      fixedContent = fixedContent.replace(fix.pattern, fix.replacement);
    });
  }
  
  return fixedContent;
}

function fixFile(filePath) {
  try {
    const content = fs.readFileSync(filePath, 'utf8');
    const fixedContent = applyFixes(content, filePath);
    
    if (content !== fixedContent) {
      fs.writeFileSync(filePath, fixedContent, 'utf8');
      console.log(`Fixed: ${filePath}`);
      return true;
    }
    return false;
  } catch (error) {
    console.error(`Error processing ${filePath}:`, error.message);
    return false;
  }
}

function findTSFiles(dir) {
  const files = [];
  
  try {
    const entries = fs.readdirSync(dir, { withFileTypes: true });
    
    for (const entry of entries) {
      const fullPath = path.join(dir, entry.name);
      
      if (entry.isDirectory() && !['node_modules', '.git', 'dist', 'build'].includes(entry.name)) {
        files.push(...findTSFiles(fullPath));
      } else if (entry.isFile() && /\.(ts|tsx)$/.test(entry.name)) {
        files.push(fullPath);
      }
    }
  } catch (error) {
    console.error(`Error reading directory ${dir}:`, error.message);
  }
  
  return files;
}

// Main execution
console.log('Starting final ESLint error fixes...');

const srcDir = path.join(__dirname, '..', 'frontend', 'src');
const tsFiles = findTSFiles(srcDir);

console.log(`Found ${tsFiles.length} TypeScript files to process`);

let fixedFiles = 0;
for (const file of tsFiles) {
  if (fixFile(file)) {
    fixedFiles++;
  }
}

console.log(`\nCompleted! Fixed ${fixedFiles} files out of ${tsFiles.length} total files.`);
console.log('Run npm run lint again to verify all errors are resolved.');