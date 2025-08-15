#!/usr/bin/env node
/**
 * Final variable fixes - fix remaining variable reference issues
 */

const fs = require('fs');
const path = require('path');
const { execSync } = require('child_process');

const frontendSrc = path.join(__dirname, '..', 'frontend', 'src');

function fixVariableReferences(filePath) {
  try {
    let content = fs.readFileSync(filePath, 'utf8');
    const originalContent = content;
    
    // Fix cases where we prefixed variables with _ but still use the original name
    // Pattern: onChange={(_e) => setSomething(e.target.value)} should be onChange={(_e) => setSomething(_e.target.value)}
    content = content.replace(/onChange={\s*\(([^)]+)\)\s*=>\s*[^}]*e\.target/g, (match, param) => {
      if (param.trim().startsWith('_')) {
        return match.replace('e.target', `${param.trim()}.target`);
      }
      return match;
    });
    
    // Fix onClick={(e) => e.stopPropagation()} when parameter is _e
    content = content.replace(/onClick={\s*\(([^)]+)\)\s*=>\s*[^}]*e\.stopPropagation/g, (match, param) => {
      if (param.trim().startsWith('_')) {
        return match.replace('e.stopPropagation', `${param.trim()}.stopPropagation`);
      }
      return match;
    });
    
    // Fix onPageChange={(e, newPage) => setPage(newPage)} when parameter is (_e, newPage)
    content = content.replace(/onPageChange={\s*\(([^,]+),([^)]+)\)\s*=>\s*[^}]*setPage/g, (match, param1, param2) => {
      return match; // These are usually correct
    });
    
    // Fix onRowsPerPageChange={(e) => { setRowsPerPage(parseInt(e.target.value, 10)); setPage(0); }}
    content = content.replace(/onRowsPerPageChange={\s*\(([^)]+)\)\s*=>\s*{[^}]*parseInt\(e\.target\.value/g, (match, param) => {
      if (param.trim().startsWith('_')) {
        return match.replace('parseInt(e.target.value', `parseInt(${param.trim()}.target.value`);
      }
      return match;
    });
    
    // Fix if (_error) checks where variable should be error
    content = content.replace(/if\s*\(\s*_error\s*\)/g, 'if (error)');
    content = content.replace(/return\s*<Alert[^>]*>{error}<\/Alert>/g, (match) => {
      // Check if error variable exists in scope
      return match;
    });
    
    // Fix missing FormControl, InputLabel, Select imports
    const lines = content.split('\n');
    let hasFormControlImport = false;
    let hasInputLabelImport = false; 
    let hasSelectImport = false;
    let hasSwitchImport = false;
    let hasFormControlLabelImport = false;
    
    for (const line of lines) {
      if (line.includes('FormControl')) hasFormControlImport = true;
      if (line.includes('InputLabel')) hasInputLabelImport = true;
      if (line.includes('Select')) hasSelectImport = true;
      if (line.includes('Switch')) hasSwitchImport = true;
      if (line.includes('FormControlLabel')) hasFormControlLabelImport = true;
    }
    
    // Check if we need these imports
    const needsFormControl = content.includes('<FormControl') && !hasFormControlImport;
    const needsInputLabel = content.includes('<InputLabel') && !hasInputLabelImport;
    const needsSelect = content.includes('<Select') && !hasSelectImport;
    const needsSwitch = content.includes('<Switch') && !hasSwitchImport;
    const needsFormControlLabel = content.includes('<FormControlLabel') && !hasFormControlLabelImport;
    
    if (needsFormControl || needsInputLabel || needsSelect || needsSwitch || needsFormControlLabel) {
      // Add missing imports to existing MUI import
      const muiImportRegex = /from\s+['"]@mui\/material['"]/;
      const muiImportMatch = content.match(muiImportRegex);
      
      if (muiImportMatch) {
        let missingImports = [];
        if (needsFormControl) missingImports.push('FormControl');
        if (needsInputLabel) missingImports.push('InputLabel');
        if (needsSelect) missingImports.push('Select');
        if (needsSwitch) missingImports.push('Switch');
        if (needsFormControlLabel) missingImports.push('FormControlLabel');
        
        const importString = missingImports.join(',\n  ');
        content = content.replace(/}\s+from\s+['"]@mui\/material['"]/, `  ${importString},\n} from '@mui/material'`);
      }
    }
    
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

const problematicFiles = [
  'components/BulkOperations/EnhancedBulkOperations.tsx',
  'components/Analytics/AnalyticsDashboard.tsx',
  'components/Analytics/CompetitiveAnalysisDashboard.tsx',
  'components/Analytics/UserBehaviorDashboard.tsx',
];

console.log('🔧 Fixing variable reference issues...\n');

let fixedCount = 0;
for (const relativeFile of problematicFiles) {
  const filePath = path.join(frontendSrc, relativeFile);
  
  if (fs.existsSync(filePath)) {
    console.log(`Processing ${relativeFile}...`);
    
    if (fixVariableReferences(filePath)) {
      console.log(`✅ Fixed ${relativeFile}`);
      fixedCount++;
    } else {
      console.log(`⏭️  No changes needed for ${relativeFile}`);
    }
  }
}

console.log(`\n✨ Fixed ${fixedCount} files`);

// Check results
console.log('\n🔍 Checking ESLint results...');
try {
  execSync('cd frontend && npm run lint', { encoding: 'utf8', stdio: 'inherit' });
  console.log('✅ All ESLint issues resolved!');
} catch (error) {
  const output = error.stdout || error.message;
  if (output) {
    const lines = output.split('\n');
    const summary = lines.slice(-5).join('\n');
    console.log('📊 Final status:');
    console.log(summary);
  }
}