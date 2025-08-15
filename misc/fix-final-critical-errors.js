#!/usr/bin/env node
/**
 * Fix Final Critical Errors
 * Target remaining specific critical issues
 */

const fs = require('fs');
const path = require('path');

const srcDir = path.join(__dirname, '..', 'frontend', 'src');

// Critical fixes for specific files
const fileFixes = {
  'BatchOperations.tsx': [
    // Fix malformed condition with wrong parentheses
    {
      pattern: /\(job\.status === 'running' \|\| job\.status === 'paused'\) && \(/g,
      replacement: '((job.status === \'running\' || job.status === \'paused\') && ('
    },
    // Fix malformed onClick handler
    {
      pattern: /onClick=\{\(\(\) => handleStopJob\(job\.id\>/g,
      replacement: 'onClick={() => handleStopJob(job.id)}'
    },
    // Fix estimated cost calculation
    {
      pattern: /\$\{newBatch\.videoCount \* 2\.04\.toFixed\(2\)\}/g,
      replacement: '${(newBatch.videoCount * 2.04).toFixed(2)}'
    },
    // Fix onChange handlers with wrong variable names
    {
      pattern: /onChange=\{\(_e\) => setNewBatch\(\{ \.\.\.newBatch, name: e\.target\.value\}\)/g,
      replacement: 'onChange={(_e) => setNewBatch({ ...newBatch, name: _e.target.value })}'
    },
    {
      pattern: /onChange=\{\(_e\) => setNewBatch\(\{ \.\.\.newBatch, type: e\.target\.value\}\)/g,
      replacement: 'onChange={(_e) => setNewBatch({ ...newBatch, type: _e.target.value })}'
    },
    {
      pattern: /onChange=\{\(_e\) => setNewBatch\(\{ \.\.\.newBatch, priority: e\.target\.value\}\)/g,
      replacement: 'onChange={(_e) => setNewBatch({ ...newBatch, priority: _e.target.value })}'
    },
    {
      pattern: /onChange=\{_e\) => setNewBatch\(\{ \.\.\.newBatch, videoCount: parseInt\(e\.target\.value\)\}\)/g,
      replacement: 'onChange={(_e) => setNewBatch({ ...newBatch, videoCount: parseInt(_e.target.value) })}'
    },
    // Fix checkbox event handlers
    {
      pattern: /if \(e\.target\.checked\)/g,
      replacement: 'if (_e.target.checked)'
    },
    // Fix dialog close handlers
    {
      pattern: /onClick=\{\(\(\) => setIsCreateDialogOpen\(false\)/g,
      replacement: 'onClick={() => setIsCreateDialogOpen(false)}'
    },
    // Fix step handlers
    {
      pattern: /onClick=\{\(\(\) => setActiveStep\(activeStep - 1\)/g,
      replacement: 'onClick={() => setActiveStep(activeStep - 1)}'
    },
    {
      pattern: /onClick=\{\(\(\) => setActiveStep\(activeStep \+ 1\)/g,
      replacement: 'onClick={() => setActiveStep(activeStep + 1)}'
    }
  ],
  
  'EnhancedBulkOperations.tsx': [
    // Fix Speed Dial handlers
    {
      pattern: /onOpen=\{\) => setSpeedDialOpen\(true\)/g,
      replacement: 'onOpen={() => setSpeedDialOpen(true)}'
    },
    {
      pattern: /onClose=\{\(\) => setSpeedDialOpen\(false\)/g,
      replacement: 'onClose={() => setSpeedDialOpen(false)}'
    },
    // Fix event parameter name inconsistency
    {
      pattern: /const handleSelectRange = useCallback\(\(startId: string, endId: string, _event: React\.MouseEvent\) => \{[\s\S]*?if \(!event\.shiftKey\) return;/g,
      replacement: (match) => match.replace('!event.shiftKey', '!_event.shiftKey')
    }
  ]
};

function processFile(filePath) {
  try {
    let content = fs.readFileSync(filePath, 'utf8');
    const fileName = path.basename(filePath);
    let hasChanges = false;

    // Apply file-specific fixes
    if (fileFixes[fileName]) {
      fileFixes[fileName].forEach(fix => {
        const oldContent = content;
        if (typeof fix.replacement === 'function') {
          content = content.replace(fix.pattern, fix.replacement);
        } else {
          content = content.replace(fix.pattern, fix.replacement);
        }
        if (content !== oldContent) {
          hasChanges = true;
          console.log(`✓ Applied final fix in ${fileName}: ${fix.pattern.toString().substring(0, 50)}...`);
        }
      });
    }

    if (hasChanges) {
      fs.writeFileSync(filePath, content);
      console.log(`✅ Fixed final critical errors in ${fileName}`);
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

console.log('🔧 Starting final critical error fixes...');
walkDir(srcDir);
console.log('✅ Final critical error fixes completed!');