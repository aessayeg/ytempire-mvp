#!/usr/bin/env node
/**
 * Fix Remaining Syntax Errors - Second Pass
 * Targets remaining specific syntax issues
 */

const fs = require('fs');
const path = require('path');

const srcDir = path.join(__dirname, '..', 'frontend', 'src');

// Remaining syntax error patterns
const fixes = [
  // Fix double closing brace in AnalyticsDashboard
  {
    pattern: /\$\{.*\.toFixed\(2\)\}\}/g,
    replacement: (match) => match.replace('}}', '}')
  },

  // Fix double closing brace in CompetitiveAnalysisDashboard
  {
    pattern: /onChange=\{.*\}\}/g,
    replacement: (match) => match.replace('}}', '}')
  },

  // Fix malformed condition in BatchOperations
  {
    pattern: /\(job\.status === 'running' \|\| job\.status === 'paused'\) &&/g,
    replacement: '(job.status === \'running\' || job.status === \'paused\') && ('
  },

  // Fix malformed onClick handlers with missing closing parentheses
  {
    pattern: /onClick=\{\(\(\) => handleStopJob\(job\.id\)/g,
    replacement: 'onClick={() => handleStopJob(job.id)}'
  },

  // Fix missing opening parenthesis for Speed Dial onOpen
  {
    pattern: /onOpen=\{\) => setSpeedDialOpen\(true\)/g,
    replacement: 'onOpen={() => setSpeedDialOpen(true)}'
  },

  // Fix missing closing parentheses for Speed Dial onClose
  {
    pattern: /onClose=\{\(\) => setSpeedDialOpen\(false\)/g,
    replacement: 'onClose={() => setSpeedDialOpen(false)}'
  },

  // Fix variable name mismatch in event handlers (e vs _e)
  {
    pattern: /onChange=\{\(_e\) => setNewBatch\(\{ \.\.\.newBatch, name: e\.target\.value\}\)/g,
    replacement: 'onChange={(_e) => setNewBatch({ ...newBatch, name: _e.target.value })}'
  },

  // Fix variable name mismatch in batch operations
  {
    pattern: /onChange=\{\(_e\) => setNewBatch\(\{ \.\.\.newBatch, type: e\.target\.value\}\)/g,
    replacement: 'onChange={(_e) => setNewBatch({ ...newBatch, type: _e.target.value })}'
  },

  // Fix variable name mismatch in priority selection
  {
    pattern: /onChange=\{\(_e\) => setNewBatch\(\{ \.\.\.newBatch, priority: e\.target\.value\}\)/g,
    replacement: 'onChange={(_e) => setNewBatch({ ...newBatch, priority: _e.target.value })}'
  },

  // Fix variable name mismatch in video count
  {
    pattern: /onChange=\{_e\) => setNewBatch\(\{ \.\.\.newBatch, videoCount: parseInt\(e\.target\.value\)\}\)/g,
    replacement: 'onChange={(_e) => setNewBatch({ ...newBatch, videoCount: parseInt(_e.target.value) })}'
  },

  // Fix variable name mismatch in checkbox handlers
  {
    pattern: /onChange=\{\(_e\) => setNewBatch\(\{[\s\S]*?options: \{ \.\.\.newBatch\.options, \w+: e\.target\.checked \}/g,
    replacement: (match) => match.replace('e.target.checked', '_e.target.checked')
  },

  // Fix variable name mismatch in channel selection
  {
    pattern: /onChange=\{\(_e\) => \{[\s\S]*?if \(e\.target\.checked\)/g,
    replacement: (match) => match.replace('e.target.checked', '_e.target.checked')
  },

  // Fix variable name mismatch in select all checkbox
  {
    pattern: /onChange=\{\(_e\) => \{[\s\S]*?if \(e\.target\.checked\) \{/g,
    replacement: (match) => match.replace('e.target.checked', '_e.target.checked')
  }
];

// File-specific fixes
const specificFixes = {
  'AnalyticsDashboard.tsx': [
    // Fix duplicate MenuItem import
    {
      pattern: /MenuItem,\n\s*MenuItem,/g,
      replacement: 'MenuItem,'
    },
    // Add missing imports
    {
      pattern: /import \{[\s\S]*?} from '@mui\/material';\n/,
      replacement: (match) => {
        if (!match.includes('ThumbUp')) {
          return match.replace('} from \'@mui/material\';', '  ThumbUp\n} from \'@mui/material\';');
        }
        return match;
      }
    }
  ],
  
  'CompetitiveAnalysisDashboard.tsx': [
    // Fix duplicate MenuItem import
    {
      pattern: /MenuItem,\n\s*MenuItem,/g,
      replacement: 'MenuItem,'
    },
    // Add missing imports
    {
      pattern: /} from '@mui\/icons-material';\n/,
      replacement: (match) => {
        if (!match.includes('Stack')) {
          return `  Stack\n} from '@mui/material';\n` + 
                 `import {\n` +
                 `  TrendingUp as TrendingUpIcon,\n` +
                 `  TrendingDown as TrendingDownIcon,\n` +
                 `  Group as GroupIcon,\n` +
                 `  Speed as SpeedIcon,\n` +
                 `  Info as InfoIcon,\n` +
                 `  Download as DownloadIcon,\n` +
                 `  Label as LabelIcon,\n` +
                 `  Copy as CopyIcon,\n` +
                 `  Analytics as AnalyticsIcon,\n` +
                 `  Badge,\n` +
                 `  Tooltip,\n` +
                 `  MoreVert as MoreVertIcon,\n` +
                 `  Star as StarIcon,\n` +
                 `  Folder as FolderIcon,\n` +
                 `  Video as VideoIcon,\n` +
                 `  Image as ImageIcon,\n` +
                 `  InsertDriveFile as FileIcon,\n` +
                 `  TableChart as TableIcon,\n` +
                 `  GridView as GridIcon,\n` +
                 `  SelectAll as SelectAllIcon,\n` +
                 `  Deselect as DeselectAllIcon\n` +
                 match;
        }
        return match;
      }
    }
  ],

  'EnhancedBulkOperations.tsx': [
    // Add missing imports
    {
      pattern: /} from '@mui\/icons-material';\n/,
      replacement: (match) => {
        if (!match.includes('MoreVert')) {
          return match.replace('} from \'@mui/icons-material\';', 
                              '  MoreVert as MoreVertIcon,\n' +
                              '  Star as StarIcon,\n' +
                              '  Folder as FolderIcon,\n' +
                              '  Video as VideoIcon,\n' +
                              '  Image as ImageIcon,\n' +
                              '  InsertDriveFile as FileIcon,\n' +
                              '  TableChart as TableIcon,\n' +
                              '  GridView as GridIcon,\n' +
                              '  SelectAll as SelectAllIcon,\n' +
                              '  Deselect as DeselectAllIcon,\n' +
                              '  Download as DownloadIcon,\n' +
                              '  Label as LabelIcon,\n' +
                              '  Copy as CopyIcon\n' +
                              '} from \'@mui/icons-material\';');
        }
        return match;
      }
    },
    // Add missing imports from material
    {
      pattern: /} from '@mui\/material';\n/,
      replacement: (match) => {
        if (!match.includes('Stack')) {
          return match.replace('} from \'@mui/material\';', 
                              '  Stack,\n' +
                              '  Tooltip\n' +
                              '} from \'@mui/material\';');
        }
        return match;
      }
    }
  ],

  'BatchOperations.tsx': [
    // Add missing imports
    {
      pattern: /} from '@mui\/icons-material';\n/,
      replacement: (match) => {
        if (!match.includes('PlayArrow')) {
          return match.replace('} from \'@mui/icons-material\';',
                              '  PlayArrow as PlayIcon,\n' +
                              '  Pause as PauseIcon,\n' +
                              '  Stop as StopIcon,\n' +
                              '  Schedule as ScheduleIcon,\n' +
                              '  CheckCircle as CheckCircleIcon,\n' +
                              '  Error as ErrorIcon,\n' +
                              '  Speed as SpeedIcon,\n' +
                              '  Queue as QueueIcon,\n' +
                              '  VideoFile as VideoIcon,\n' +
                              '  CircularProgress,\n' +
                              '  Stepper,\n' +
                              '  Step,\n' +
                              '  StepLabel,\n' +
                              '  List,\n' +
                              '  ListItem,\n' +
                              '  ListItemIcon,\n' +
                              '  ListItemText\n' +
                              '} from \'@mui/icons-material\';');
        }
        return match;
      }
    },
    // Add missing Grid import from mui/material
    {
      pattern: /} from '@mui\/material';\n/,
      replacement: (match) => {
        if (!match.includes('Grid')) {
          return match.replace('} from \'@mui/material\';', 
                              '  Grid\n' +
                              '} from \'@mui/material\';');
        }
        return match;
      }
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
      if (typeof fix.replacement === 'function') {
        content = content.replace(fix.pattern, fix.replacement);
      } else {
        content = content.replace(fix.pattern, fix.replacement);
      }
      if (content !== oldContent) {
        hasChanges = true;
        console.log(`✓ Applied fix in ${fileName}: ${fix.pattern.toString()}`);
      }
    });

    // Apply file-specific fixes
    if (specificFixes[fileName]) {
      specificFixes[fileName].forEach(fix => {
        const oldContent = content;
        if (typeof fix.replacement === 'function') {
          content = content.replace(fix.pattern, fix.replacement);
        } else {
          content = content.replace(fix.pattern, fix.replacement);
        }
        if (content !== oldContent) {
          hasChanges = true;
          console.log(`✓ Applied specific fix in ${fileName}`);
        }
      });
    }

    if (hasChanges) {
      fs.writeFileSync(filePath, content);
      console.log(`✅ Fixed remaining errors in ${fileName}`);
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

console.log('🔧 Starting remaining syntax error fixes...');
walkDir(srcDir);
console.log('✅ Remaining syntax error fixes completed!');