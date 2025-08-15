const fs = require('fs');
const path = require('path');

// Define fix patterns for specific error types
const fixes = [
  // Fix variable reference mismatches
  {
    pattern: /console\.error\(`Error in event listener for "\${event}":`, error\);/g,
    replacement: 'console.error(`Error in event listener for "${event}":`, _);'
  },
  
  // Fix unterminated template literals and other parsing issues
  {
    pattern: /\r`/g,
    replacement: '`'
  },
  
  // Fix malformed object properties (trailing commas in wrong places)
  {
    pattern: /,(\s*}\s*)/g,
    replacement: '$1'
  },
  
  // Fix missing opening braces for arrow functions
  {
    pattern: /=>\s*{([^}]*)\s*}\s*;\s*}/g,
    replacement: '=> {\n    $1\n  }'
  },
  
  // Fix malformed arrow function syntax
  {
    pattern: /\(\(\(/g,
    replacement: '(('
  },
  
  // Fix spacing in pixel values
  {
    pattern: /'9999 px'/g,
    replacement: "'9999px'"
  },
  
  // Fix malformed const declarations
  {
    pattern: /const,\s*(\w+)/g,
    replacement: 'const $1'
  },
  
  // Fix malformed export declarations
  {
    pattern: /export const,\s*(\w+)/g,
    replacement: 'export const $1'
  },
  
  // Fix template literal syntax errors
  {
    pattern: /\`([^`]*);}`/g,
    replacement: '`$1`'
  },
  
  // Fix incomplete parentheses in function calls
  {
    pattern: /onClick={([^}]+)}/g,
    replacement: function(match, p1) {
      // Count parentheses to ensure they're balanced
      const openParens = (p1.match(/\(/g) || []).length;
      const closeParens = (p1.match(/\)/g) || []).length;
      if (openParens > closeParens) {
        return `onClick={${p1}}`;
      }
      return match;
    }
  },
  
  // Fix malformed filter syntax
  {
    pattern: /\.filter\(item => \{([^}]*)\s*}\s*\)/g,
    replacement: '.filter(item => $1)'
  },
  
  // Fix speed dial handlers
  {
    pattern: /onOpen={\) =>/g,
    replacement: 'onOpen={() =>'
  },
  
  // Fix malformed object destructuring
  {
    pattern: /\[\s*{([^}]+)}\s*\]/g,
    replacement: '[{$1}]'
  },
  
  // Fix missing spaces in style values
  {
    pattern: /'(\d+) px'/g,
    replacement: "'$1px'"
  },
  
  // Fix numeric literal issues
  {
    pattern: /(\d+) (\w+)/g,
    replacement: function(match, num, word) {
      // Only fix if it looks like a CSS value
      if (['px', 's', 'ms', '%'].includes(word)) {
        return `${num}${word}`;
      }
      return match;
    }
  },
  
  // Fix malformed type assertions
  {
    pattern: /as,\s*(\w+)/g,
    replacement: 'as $1'
  },
  
  // Fix duplicate imports
  {
    pattern: /import\s*{\s*([^}]+),\s*([^}]+)\s*}/g,
    replacement: function(match, p1, p2) {
      // Remove duplicate MenuItem import
      if (p1.includes('MenuItem') && p2.includes('MenuItem')) {
        const items = [...new Set([...p1.split(','), ...p2.split(',')].map(item => item.trim()))];
        return `import { ${items.join(', ')} }`;
      }
      return match;
    }
  },
  
  // Fix malformed stepper imports and missing components
  {
    pattern: /import\s*{\s*([^}]*)\s*}\s*from\s*'@mui\/material';/g,
    replacement: function(match, imports) {
      const importList = imports.split(',').map(imp => imp.trim()).filter(Boolean);
      // Add missing imports if they're used in stepper
      if (match.includes('Stepper')) {
        const requiredImports = ['Stepper', 'Step', 'StepLabel'];
        requiredImports.forEach(imp => {
          if (!importList.includes(imp)) {
            importList.push(imp);
          }
        });
      }
      return `import {\n  ${importList.join(',\n  ')}\n} from '@mui/material';`;
    }
  }
];

// File-specific fixes
const fileSpecificFixes = {
  'EventEmitter.ts': [
    {
      pattern: /console\.error\(`Error in event listener for "\${event}":`, error\);/,
      replacement: 'console.error(`Error in event listener for "${event}":`, _);'
    }
  ],
  
  'AnalyticsDashboard.tsx': [
    {
      pattern: /const days = timeRange === '7 d' \? 7 : timeRange === '30 d' \? 30 : 90;\s*const revenue = Array\.from\(\{ length: days \}, \(_, i\) => \{ const date = subDays\(new Date\(\), days - 1 - i\);/,
      replacement: `const days = timeRange === '7 d' ? 7 : timeRange === '30 d' ? 30 : 90;
    const revenue = Array.from({ length: days }, (_, i) => {
      const date = subDays(new Date(), days - 1 - i);`
    }
  ],
  
  'BatchOperations.tsx': [
    {
      pattern: /setJobs\(prevJobs => \{\}/,
      replacement: 'setJobs(prevJobs =>'
    },
    {
      pattern: /handleStartJob = \(jobId: string\) => \{\s*setJobs\(jobs\.map\(job => \{\}/,
      replacement: `handleStartJob = (jobId: string) => {
    setJobs(jobs.map(job =>`
    }
  ]
};

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
  if (fileSpecificFixes[baseName]) {
    fileSpecificFixes[baseName].forEach(fix => {
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
console.log('Starting ESLint error fixes v2...');

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