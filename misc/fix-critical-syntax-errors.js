#!/usr/bin/env node
/**
 * Fix Critical Syntax Errors - Final ESLint Cleanup
 * Targets specific malformed handlers and syntax issues found in the remaining 144 issues
 */

const fs = require('fs');
const path = require('path');

const srcDir = path.join(__dirname, '..', 'frontend', 'src');

// Critical syntax error patterns to fix
const fixes = [
  // Fix malformed tab handlers like `onChange={(e, v) => setCurrentTab(v}` (missing closing parenthesis)
  {
    pattern: /onChange=\{\(e, v\) => setCurrentTab\(v\}/g,
    replacement: 'onChange={(e, v) => setCurrentTab(v)}'
  },
  
  // Fix malformed event handlers like `onChange={(e, v) => v && setChartType(v)}` (missing closing parenthesis)
  {
    pattern: /onChange=\{\(\(e, v\) => v && setChartType\(v\)\}/g,
    replacement: 'onChange={((e, v) => v && setChartType(v))}'
  },
  
  // Fix malformed toggle button handlers like `onChange={((e, v) => v && setViewMode(v)}`
  {
    pattern: /onChange=\{\(\(e, v\) => v && setViewMode\(v\)\}/g,
    replacement: 'onChange={((e, v) => v && setViewMode(v))}'
  },

  // Fix malformed onChange handlers with missing closing parenthesis
  {
    pattern: /onChange=\{\(_e\) => setTimeRange\(_e\.target\.value\)/g,
    replacement: 'onChange={(_e) => setTimeRange(_e.target.value)}'
  },

  // Fix malformed onClick handlers with extra opening parenthesis
  {
    pattern: /onClick=\{\(\(\) => handleOperation\(op\)/g,
    replacement: 'onClick={() => handleOperation(op)}'
  },

  // Fix malformed onClick handlers with missing closing parenthesis
  {
    pattern: /onClick=\{\(\) => setSelectedIds\(new Set\(\)\)/g,
    replacement: 'onClick={() => setSelectedIds(new Set())}'
  },

  // Fix malformed onClick handlers in bulk operations
  {
    pattern: /onClick=\{\(\(\) => removeCompetitor\(competitor\.id\)/g,
    replacement: 'onClick={() => removeCompetitor(competitor.id)}'
  },

  // Fix malformed onClick handlers for select item
  {
    pattern: /onClick=\{\(\) => handleSelectItem\(item\.id\)/g,
    replacement: 'onClick={() => handleSelectItem(item.id)}'
  },

  // Fix malformed onChange with missing closing parenthesis for checkboxes
  {
    pattern: /onChange=\{\(\) => toggleCompetitorSelection\(competitor\.id\)/g,
    replacement: 'onChange={() => toggleCompetitorSelection(competitor.id)}'
  },

  // Fix malformed onChange handlers for checkboxes
  {
    pattern: /onChange=\{\(\) => handleSelectItem\(item\.id\)/g,
    replacement: 'onChange={() => handleSelectItem(item.id)}'
  },

  // Fix malformed onClick in table rows with range selection
  {
    pattern: /onClick=\{\(_e\) => handleSelectRange\(\n\s+index > 0 \? paginatedItems\[index - 1\]\.id : item\.id,\n\s+item\.id,\n\s+_e\n\s+\)/g,
    replacement: 'onClick={(_e) => handleSelectRange(\n                index > 0 ? paginatedItems[index - 1].id : item.id,\n                item.id,\n                _e\n              )}'
  },

  // Fix malformed template literals with missing parenthesis
  {
    pattern: /\$\{\(totalRevenue \/ revenueData\.length\.toFixed\(2\)/g,
    replacement: '${(totalRevenue / revenueData.length).toFixed(2)}'
  },

  // Fix malformed template literals with missing parenthesis for subscriber counts
  {
    pattern: /\{\(competitor\.subscriberCount \/ 1000000\)\.toFixed\(2\}M/g,
    replacement: '{(competitor.subscriberCount / 1000000).toFixed(2)}M'
  },

  // Fix malformed template literals for view counts
  {
    pattern: /\{\(competitor\.viewCount \/ 1000000\)\.toFixed\(1\}M/g,
    replacement: '{(competitor.viewCount / 1000000).toFixed(1)}M'
  },

  // Fix malformed template literals for revenue
  {
    pattern: /\$\{\(item\.value\.toFixed\(2\)\}/g,
    replacement: '${item.value.toFixed(2)}'
  },

  // Fix malformed onPageChange handlers
  {
    pattern: /onPageChange=\{e, newPage\) => setPage\(newPage\)/g,
    replacement: 'onPageChange={(e, newPage) => setPage(newPage)}'
  },

  // Fix malformed onChange handlers for filter selection
  {
    pattern: /onChange=\{\(_e\) => setFilterType\(_e\.target\.value as string\)/g,
    replacement: 'onChange={(_e) => setFilterType(_e.target.value as string)}'
  },

  // Fix malformed onChange handlers for sort selection
  {
    pattern: /onChange=\{\(_e\) => setSortBy\(_e\.target\.value as string\)/g,
    replacement: 'onChange={(_e) => setSortBy(_e.target.value as string)}'
  },

  // Fix malformed onClick handlers for speed dial
  {
    pattern: /onOpen=\{\) => setSpeedDialOpen\(true\)/g,
    replacement: 'onOpen={() => setSpeedDialOpen(true)}'
  },

  // Fix malformed onClose handlers for speed dial
  {
    pattern: /onClose=\{\(\) => setSpeedDialOpen\(false\)/g,
    replacement: 'onClose={() => setSpeedDialOpen(false)}'
  },

  // Fix malformed onClick handlers in speed dial actions
  {
    pattern: /onClick=\{\(\(\) => \{/g,
    replacement: 'onClick={() => {'
  },

  // Fix malformed onClick handlers for confirm dialog
  {
    pattern: /onClick=\{\(\(\) => setConfirmDialog\(\{ open: false \}\)\}/g,
    replacement: 'onClick={() => setConfirmDialog({ open: false })}'
  },

  // Fix malformed onClick handlers for batch operations
  {
    pattern: /onClick=\{\(\(\) => setIsCreateDialogOpen\(true\)\}/g,
    replacement: 'onClick={() => setIsCreateDialogOpen(true)}'
  },

  // Fix malformed onClick handlers for job actions
  {
    pattern: /onClick=\{\(\(\) => handleStartJob\(job\.id\)\}/g,
    replacement: 'onClick={() => handleStartJob(job.id)}'
  },

  // Fix malformed onClick handlers for job pause
  {
    pattern: /onClick=\{\(\(\) => handlePauseJob\(job\.id\)\}/g,
    replacement: 'onClick={() => handlePauseJob(job.id)}'
  },

  // Fix malformed onClick handlers for job stop
  {
    pattern: /onClick=\{\(\(\) => handleStopJob\(job\.id\)\}/g,
    replacement: 'onClick={() => handleStopJob(job.id)}'
  },

  // Fix malformed onClick handlers for job delete
  {
    pattern: /onClick=\{\(\(\) => handleDeleteJob\(job\.id\)\}/g,
    replacement: 'onClick={() => handleDeleteJob(job.id)}'
  },

  // Fix malformed onChange handlers for new batch
  {
    pattern: /onChange=\{\(_e\) => setNewBatch\(\{ \.\.\.newBatch, name: e\.target\.value\}\)\}/g,
    replacement: 'onChange={(_e) => setNewBatch({ ...newBatch, name: _e.target.value })}'
  },

  // Fix malformed onChange handlers for batch type
  {
    pattern: /onChange=\{\(_e\) => setNewBatch\(\{ \.\.\.newBatch, type: e\.target\.value\}\)\}/g,
    replacement: 'onChange={(_e) => setNewBatch({ ...newBatch, type: _e.target.value })}'
  },

  // Fix malformed onChange handlers for batch priority
  {
    pattern: /onChange=\{\(_e\) => setNewBatch\(\{ \.\.\.newBatch, priority: e\.target\.value\}\)\}/g,
    replacement: 'onChange={(_e) => setNewBatch({ ...newBatch, priority: _e.target.value })}'
  },

  // Fix malformed onClick handlers for dialog close
  {
    pattern: /onClick=\{\(\(\) => setIsCreateDialogOpen\(false\)/g,
    replacement: 'onClick={() => setIsCreateDialogOpen(false)}'
  },

  // Fix malformed onClick handlers for stepper back
  {
    pattern: /onClick=\{\(\(\) => setActiveStep\(activeStep - 1\)/g,
    replacement: 'onClick={() => setActiveStep(activeStep - 1)}'
  },

  // Fix malformed onClick handlers for stepper next
  {
    pattern: /onClick=\{\(\(\) => setActiveStep\(activeStep \+ 1\)/g,
    replacement: 'onClick={() => setActiveStep(activeStep + 1)}'
  },

  // Fix malformed condition syntax
  {
    pattern: /\(job\.status === 'running' \|\| job\.status === 'paused'\) &&/g,
    replacement: '(job.status === \'running\' || job.status === \'paused\') && ('
  },

  // Fix malformed estimated cost calculation
  {
    pattern: /\$\{newBatch\.videoCount \* 2\.04\.toFixed\(2\)\}/g,
    replacement: '${(newBatch.videoCount * 2.04).toFixed(2)}'
  },

  // Fix malformed map function with missing opening parenthesis
  {
    pattern: /\{links\.map\(\(link\) => \(_<Link/g,
    replacement: '{links.map((link) => (\n        <Link'
  },

  // Fix malformed onClick handlers in accessibility component
  {
    pattern: /onClick=\{\(\(e\) => \{/g,
    replacement: 'onClick={(e) => {'
  }
];

// Additional specific fixes for individual files
const fileSpecificFixes = {
  'CompetitiveAnalysisDashboard.tsx': [
    // Fix duplicate imports
    {
      pattern: /,\n\s*Add as AddIcon,\n\s*Delete as DeleteIcon/g,
      replacement: ''
    },
    // Fix duplicate MenuItem imports
    {
      pattern: /MenuItem\s*,\n\s*MenuItem,/g,
      replacement: 'MenuItem,'
    }
  ],
  
  'EnhancedBulkOperations.tsx': [
    // Fix duplicate imports
    {
      pattern: /,\n\s*Delete as DeleteIcon,\n\s*Edit as EditIcon,\n\s*Search as SearchIcon,\n\s*Clear as ClearIcon/g,
      replacement: ''
    },
    // Fix duplicate MenuItem imports
    {
      pattern: /MenuItem\s*,\n\s*MenuItem,/g,
      replacement: 'MenuItem,'
    }
  ],

  'AnalyticsDashboard.tsx': [
    // Fix missing imports
    {
      pattern: /import \{/,
      replacement: 'import {\n  FormControl,\n  InputLabel,\n  Select,\n  MenuItem,\n  FormControlLabel,\n  Switch,\n  Tooltip as RechartsTooltip,'
    }
  ]
};

function processFile(filePath) {
  try {
    let content = fs.readFileSync(filePath, 'utf8');
    const fileName = path.basename(filePath);
    let hasChanges = false;

    // Apply general fixes
    fixes.forEach(fix => {
      const oldContent = content;
      content = content.replace(fix.pattern, fix.replacement);
      if (content !== oldContent) {
        hasChanges = true;
        console.log(`✓ Applied fix in ${fileName}: ${fix.pattern.toString()}`);
      }
    });

    // Apply file-specific fixes
    if (fileSpecificFixes[fileName]) {
      fileSpecificFixes[fileName].forEach(fix => {
        const oldContent = content;
        content = content.replace(fix.pattern, fix.replacement);
        if (content !== oldContent) {
          hasChanges = true;
          console.log(`✓ Applied file-specific fix in ${fileName}`);
        }
      });
    }

    if (hasChanges) {
      fs.writeFileSync(filePath, content);
      console.log(`✅ Fixed syntax errors in ${fileName}`);
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

console.log('🔧 Starting critical syntax error fixes...');
walkDir(srcDir);
console.log('✅ Critical syntax error fixes completed!');