#!/usr/bin/env node
/**
 * Fix completely broken import statements that were corrupted by previous scripts
 */

const fs = require('fs');
const path = require('path');
const { execSync } = require('child_process');

const frontendSrc = path.join(__dirname, '..', 'frontend', 'src');

function fixBrokenFile(filePath) {
  try {
    let content = fs.readFileSync(filePath, 'utf8');
    const lines = content.split('\n');
    const fixedLines = [];
    
    let i = 0;
    let inBrokenImportSection = false;
    let importItems = [];
    let currentModule = '';
    
    while (i < lines.length) {
      const line = lines[i];
      
      // Check if this is a completely broken import like:
      // import { useAuthStore } from '../../stores/authStore'; Box,
      if (line.match(/^import.*from.*['"];.*,/)) {
        // Extract the core import
        const match = line.match(/^(import.*from.*['"];)(.*)/);
        if (match) {
          const coreImport = match[1];
          const remainingItems = match[2].trim().replace(/^,?\s*/, '').replace(/,$/, '');
          
          fixedLines.push(coreImport);
          
          if (remainingItems) {
            importItems.push(remainingItems);
            inBrokenImportSection = true;
          }
        }
      }
      // Check for repeated broken imports
      else if (line.match(/^import.*from.*['"];.*[A-Za-z]/)) {
        // This is another broken import - extract items
        const match = line.match(/;(.*)$/);
        if (match) {
          const items = match[1].trim().replace(/^,?\s*/, '').replace(/,$/, '');
          if (items) {
            importItems.push(items);
          }
        }
      }
      // Check if we're in a section of just hanging identifiers
      else if (inBrokenImportSection && line.trim() && 
               line.trim().match(/^[A-Za-z][A-Za-z0-9]*,?$/) &&
               !line.includes('=') && 
               !line.includes('(') && 
               !line.includes('{') &&
               !line.includes('const') &&
               !line.includes('let') &&
               !line.includes('var') &&
               !line.includes('function') &&
               !line.includes('class') &&
               !line.includes('interface') &&
               !line.includes('type')) {
        importItems.push(line.trim().replace(/,$/, ''));
      }
      // Check for end of broken import section
      else if (inBrokenImportSection && 
               (line.trim() === '' ||
                line.includes('interface') ||
                line.includes('const') ||
                line.includes('export') ||
                line.includes('function') ||
                line.includes('class') ||
                line.includes('//'))) {
        
        // End the broken import section and create proper import
        if (importItems.length > 0) {
          const cleanItems = importItems.join(', ').split(',').map(s => s.trim()).filter(s => s);
          if (cleanItems.length > 0) {
            // Find the most appropriate import source
            let importSource = '@mui/material';
            if (cleanItems.some(item => item.includes('Icon'))) {
              importSource = '@mui/icons-material';
            }
            
            fixedLines.push(`import { ${cleanItems.join(', ')} } from '${importSource}';`);
          }
        }
        
        importItems = [];
        inBrokenImportSection = false;
        fixedLines.push(line);
      }
      // Regular line
      else if (!inBrokenImportSection) {
        fixedLines.push(line);
      }
      
      i++;
    }
    
    // Handle any remaining items
    if (importItems.length > 0) {
      const cleanItems = importItems.join(', ').split(',').map(s => s.trim()).filter(s => s);
      if (cleanItems.length > 0) {
        fixedLines.push(`import { ${cleanItems.join(', ')} } from '@mui/material';`);
      }
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

// Get files with the most severe import issues
const brokenFiles = [
  'components/Analytics/AnalyticsDashboard.tsx',
  'components/Analytics/CompetitiveAnalysisDashboard.tsx', 
  'components/Analytics/UserBehaviorDashboard.tsx',
  'components/Auth/EmailVerification.tsx',
  'components/Auth/ForgotPasswordForm.tsx',
  'components/Auth/LoginForm.tsx',
  'components/Auth/RegisterForm.tsx',
  'components/Auth/TwoFactorAuth.tsx',
  'components/BatchOperations/BatchOperations.tsx',
  'components/BulkOperations/EnhancedBulkOperations.tsx'
];

console.log('Fixing broken import statements...');
console.log('-'.repeat(50));

let totalFixed = 0;

brokenFiles.forEach(relPath => {
  const filePath = path.join(frontendSrc, relPath);
  if (fs.existsSync(filePath) && fixBrokenFile(filePath)) {
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
  const output = error.stdout || error.message;
  const lines = output.split('\n');
  console.log(lines.slice(-10).join('\n'));
}