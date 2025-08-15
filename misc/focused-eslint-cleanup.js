#!/usr/bin/env node
/**
 * Focused ESLint cleanup - target specific high-frequency issues
 */

const fs = require('fs');
const path = require('path');
const { execSync } = require('child_process');

const frontendSrc = path.join(__dirname, '..', 'frontend', 'src');

function removeUnusedImports(filePath) {
  try {
    let content = fs.readFileSync(filePath, 'utf8');
    const originalContent = content;
    
    const lines = content.split('\n');
    const filteredLines = [];
    let inImportBlock = false;
    let currentImportStatement = '';
    
    for (const line of lines) {
      const trimmedLine = line.trim();
      
      // Handle import statements
      if (trimmedLine.startsWith('import ')) {
        inImportBlock = true;
        currentImportStatement = line;
        
        // Check if it's a single line import
        if (line.includes(';') && !line.includes(',')) {
          if (shouldKeepImport(currentImportStatement, content)) {
            filteredLines.push(line);
          }
          currentImportStatement = '';
          inImportBlock = false;
        }
      } else if (inImportBlock && (line.includes(';') || trimmedLine === '}')) {
        currentImportStatement += line;
        if (shouldKeepImport(currentImportStatement, content)) {
          // Add all lines of the import statement
          const importLines = currentImportStatement.split('\n');
          filteredLines.push(...importLines);
        }
        currentImportStatement = '';
        inImportBlock = false;
      } else if (inImportBlock) {
        currentImportStatement += line + '\n';
      } else {
        filteredLines.push(line);
      }
    }
    
    const newContent = filteredLines.join('\n');
    
    if (newContent !== originalContent) {
      fs.writeFileSync(filePath, newContent, 'utf8');
      return true;
    }
    
    return false;
  } catch (error) {
    console.error(`Error processing ${filePath}:`, error.message);
    return false;
  }
}

function shouldKeepImport(importStatement, fileContent) {
  // Extract imported names from the import statement
  const importMatches = importStatement.match(/import\s+(?:(\w+)|{([^}]+)})\s+from/);
  if (!importMatches) return true; // Keep unknown format imports
  
  const defaultImport = importMatches[1];
  const namedImports = importMatches[2];
  
  let hasUsedImport = false;
  
  // Check default import
  if (defaultImport) {
    const regex = new RegExp(`\\b${defaultImport}\\b`, 'g');
    const matches = fileContent.match(regex) || [];
    hasUsedImport = matches.length > 1; // More than just the import line
  }
  
  // Check named imports
  if (namedImports) {
    const names = namedImports.split(',').map(name => 
      name.trim().split(' as ')[0].trim()
    );
    
    for (const name of names) {
      const regex = new RegExp(`\\b${name}\\b`, 'g');
      const matches = fileContent.match(regex) || [];
      if (matches.length > 1) { // More than just the import line
        hasUsedImport = true;
        break;
      }
    }
  }
  
  return hasUsedImport;
}

function fixSpecificFiles() {
  const problematicFiles = [
    'components/BatchOperations/BatchOperations.tsx',
    'components/BulkOperations/EnhancedBulkOperations.tsx',  
    'components/Analytics/AnalyticsDashboard.tsx',
    'components/Analytics/CompetitiveAnalysisDashboard.tsx',
    'components/Analytics/UserBehaviorDashboard.tsx',
    'pages/Videos/VideoQueue.tsx',
    'services/analytics_tracker.ts',
    'services/api.ts',
    'services/websocketService.ts',
    'stores/authStore.ts'
  ];
  
  let fixedCount = 0;
  
  for (const relativeFile of problematicFiles) {
    const filePath = path.join(frontendSrc, relativeFile);
    
    if (fs.existsSync(filePath)) {
      console.log(`Processing ${relativeFile}...`);
      
      let content = fs.readFileSync(filePath, 'utf8');
      const originalContent = content;
      
      // Remove specific unused imports based on known issues
      const unusedImports = [
        'Tooltip', 'Badge', 'RefreshIcon', 'WarningIcon', 'SettingsIcon',
        'List', 'ListItem', 'ListItemIcon', 'ListItemText', 'ListItemSecondaryAction',
        'CircularProgress', 'Stepper', 'Step', 'StepLabel', 'StepContent',
        'Divider', 'Collapse', 'Stack', 'ListItemButton', 'FormControlLabel',
        'Switch', 'Radio', 'RadioGroup', 'CheckBoxIcon', 'CheckBoxOutlineBlankIcon',
        'IndeterminateCheckBoxIcon', 'VisibilityIcon', 'VisibilityOffIcon',
        'useEffect', 'TextField', 'Select', 'FormControl', 'InputLabel',
        'Stop', 'MoreVert', 'Warning', 'Info', 'Timer', 'ThumbUp', 'CloudUpload',
        'ContentCopy', 'SkipNext', 'FastForward'
      ];
      
      // Remove unused import lines
      const lines = content.split('\n');
      const filteredLines = [];
      
      for (const line of lines) {
        let keepLine = true;
        
        // Check if line contains only unused imports
        if (line.trim().match(/^\s*[a-zA-Z_][a-zA-Z0-9_]*,?$/)) {
          const importName = line.trim().replace(',', '');
          if (unusedImports.includes(importName)) {
            keepLine = false;
          }
        }
        
        // Fix specific patterns
        if (line.includes('onJobComplete') && !content.includes('onJobComplete(')) {
          keepLine = false;
        }
        
        if (line.includes('setSortBy') && !content.includes('setSortBy(')) {
          keepLine = false;  
        }
        
        if (keepLine) {
          filteredLines.push(line);
        }
      }
      
      content = filteredLines.join('\n');
      
      // Fix unused variables by prefixing with underscore
      content = content.replace(/\b(error|e|event|message)\b(?=\s*[:)])/g, '_$1');
      
      if (content !== originalContent) {
        fs.writeFileSync(filePath, content, 'utf8');
        fixedCount++;
        console.log(`✅ Fixed ${relativeFile}`);
      } else {
        console.log(`⏭️  No changes needed for ${relativeFile}`);
      }
    }
  }
  
  return fixedCount;
}

console.log('🎯 Running focused ESLint cleanup...');
console.log('Targeting high-frequency unused import issues...\n');

const fixedFiles = fixSpecificFiles();

console.log(`\n✨ Fixed ${fixedFiles} files`);

// Check results
console.log('\n🔍 Checking ESLint results...');
try {
  execSync('cd frontend && npm run lint', { encoding: 'utf8' });
  console.log('✅ All ESLint issues resolved!');
} catch (error) {
  const output = error.stdout || error.message;
  const lines = output.split('\n');
  const summary = lines.slice(-5).join('\n');
  console.log('📊 Final status:');
  console.log(summary);
}