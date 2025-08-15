/**
 * Enhanced Bulk Operations Interface
 * Complete multi-select interface with progress tracking and batch actions
 */

import React, { useState, useCallback, useMemo, useEffect } from 'react';
import Grid from '@mui/material/Grid2';
import {
  Box,
  Card,
  CardContent,
  Typography,
  Button,
  Checkbox,
  IconButton,
  Chip,
  Dialog,
  DialogTitle,
  DialogContent,
  DialogActions,
  MenuItem,
  Alert,
  LinearProgress,
  Table,
  TableHead,
  TableBody,
  TableRow,
  TableCell,
  TableContainer,
  TablePagination,
  Paper,
  Avatar,
  Fade,
  ButtonGroup,
  ToggleButton,
  ToggleButtonGroup,
  InputAdornment,
  Snackbar,
  SpeedDial,
  SpeedDialAction,
  SpeedDialIcon,
  useTheme,
  alpha
} from '@mui/material';
import {
  CheckBox as CheckBoxIcon,
  CheckBoxOutlineBlank as CheckBoxOutlineBlankIcon,
  IndeterminateCheckBox as IndeterminateCheckBoxIcon,
  Edit as EditIcon,
  Delete as DeleteIcon,
  Schedule as ScheduleIcon,
  Visibility as VisibilityIcon,
  VisibilityOff as VisibilityOffIcon,
  ContentCopy as CopyIcon,
  Archive as ArchiveIcon,
  Unarchive as UnarchiveIcon,
  Label as LabelIcon,
  Star as StarIcon,
  FilterList as FilterIcon,
  ViewModule as GridIcon,
  ViewList as TableIcon,
  Search as SearchIcon,
  MoreVert as MoreVertIcon,
  Download as DownloadIcon,
  Upload as UploadIcon,
  Settings as SettingsIcon,
  Undo as UndoIcon,
  Redo as RedoIcon,
  Clear as ClearIcon,
  SelectAll as SelectAllIcon,
  DeselectAll as DeselectAllIcon,
  Refresh as RefreshIcon,
  FolderOpen as FolderIcon,
  VideoLibrary as VideoIcon,
  Image as ImageIcon,
  AttachFile as FileIcon
} from '@mui/icons-material';

// Types
interface BulkItem {
  id: string;
  name: string;
  type: 'channel' | 'video' | 'image' | 'file';
  status: 'active' | 'paused' | 'archived' | 'processing';
  selected?: boolean;
  thumbnail?: string;
  metadata?: Record<string, unknown>;
  tags?: string[];
  starred?: boolean;
  createdAt: Date;
  modifiedAt: Date;
}

interface BulkOperation {
  id: string;
  type: 'edit' | 'delete' | 'archive' | 'export' | 'tag' | 'schedule' | 'copy';
  name: string;
  icon: React.ReactNode;
  color?: 'primary' | 'secondary' | 'error' | 'warning' | 'info' | 'success';
  requiresConfirmation?: boolean;
  allowedTypes?: Array<BulkItem['type']>;
}

interface OperationProgress {
  operationId: string;
  totalItems: number;
  processedItems: number;
  failedItems: number;
  status: 'pending' | 'processing' | 'completed' | 'failed' | 'cancelled';
  startTime?: Date;
  endTime?: Date;
  errors?: Array<{ itemId: string; _error: string }>;
}

interface BulkOperationsProps {
  items: BulkItem[];
  onOperationComplete?: (operation: string, items: string[]) => void;
  onSelectionChange?: (selectedIds: string[]) => void;
  customOperations?: BulkOperation[];
  enableDragAndDrop?: boolean;
  enableAutoSave?: boolean;
}

export const EnhancedBulkOperations: React.FC<BulkOperationsProps> = ({
  items: initialItems,
  onOperationComplete,
  onSelectionChange,
  customOperations = [],
  enableDragAndDrop = true,
  enableAutoSave = false
}) => {
  const theme = useTheme();
  const [items, setItems] = useState<BulkItem[]>(initialItems);
  const [selectedIds, setSelectedIds] = useState<Set<string>>(new Set());
  const [searchQuery, setSearchQuery] = useState('');
  const [filterType, setFilterType] = useState<BulkItem['type'] | 'all'>('all');
  const [sortBy, setSortBy] = useState<'name' | 'date' | 'type'>('name');
  const [sortOrder, setSortOrder] = useState<'asc' | 'desc'>('asc');
  const [page, setPage] = useState(0);
  const [rowsPerPage, setRowsPerPage] = useState(10);
  const [viewMode, setViewMode] = useState<'table' | 'grid' | 'list'>('table');
  const [confirmDialog, setConfirmDialog] = useState<{
    open: boolean;
    operation?: BulkOperation;
    message?: string;
  }>({ open: false });
  const [progressDialog, setProgressDialog] = useState(false);
  const [operationProgress, setOperationProgress] = useState<OperationProgress | null>(null);
  const [snackbar, setSnackbar] = useState<{
    open: boolean;
    _message: string;
    severity: 'success' | 'error' | 'warning' | 'info';
  }>({ open: false, _message: '', severity: 'info' });
  const [speedDialOpen, setSpeedDialOpen] = useState(false);
  const [history, setHistory] = useState<Array<{ action: string; items: string[]; timestamp: Date }>>([]);
  const [historyIndex, setHistoryIndex] = useState(-1);

  // Default operations
  const defaultOperations: BulkOperation[] = [
    { id: 'edit', type: 'edit', name: 'Edit', icon: <EditIcon />, color: 'primary' },
    { id: 'delete', type: 'delete', name: 'Delete', icon: <DeleteIcon />, color: 'error', requiresConfirmation: true },
    { id: 'archive', type: 'archive', name: 'Archive', icon: <ArchiveIcon />, color: 'warning' },
    { id: 'export', type: 'export', name: 'Export', icon: <DownloadIcon />, color: 'info' },
    { id: 'tag', type: 'tag', name: 'Add Tags', icon: <LabelIcon />, color: 'secondary' },
    { id: 'schedule', type: 'schedule', name: 'Schedule', icon: <ScheduleIcon />, color: 'primary' },
    { id: 'copy', type: 'copy', name: 'Duplicate', icon: <CopyIcon />, color: 'success' }
  ];

  const operations = [...defaultOperations, ...customOperations];

  // Computed values
  const filteredItems = useMemo(() => {
    let filtered = items;

    // Apply search filter
    if (searchQuery) {
      filtered = filtered.filter(item =>
        item.name.toLowerCase().includes(searchQuery.toLowerCase()) ||
        item.tags?.some(tag => tag.toLowerCase().includes(searchQuery.toLowerCase()))
      );
    }

    // Apply type filter
    if (filterType !== 'all') {
      filtered = filtered.filter(item => item.type === filterType);
    }

    // Apply sorting
    filtered.sort((a, b) => {
      let comparison = 0;
      switch (sortBy) {
        case 'name':
          comparison = a.name.localeCompare(b.name);
          break;
        case 'date':
          comparison = a.modifiedAt.getTime() - b.modifiedAt.getTime();
          break;
        case 'type':
          comparison = a.type.localeCompare(b.type);
          break;
      }
      return sortOrder === 'asc' ? comparison : -comparison;
    });

    return filtered;
  }, [items, searchQuery, filterType, sortBy, sortOrder]);

  const paginatedItems = useMemo(() => {
    const start = page * rowsPerPage;
    return filteredItems.slice(start, start + rowsPerPage);
  }, [filteredItems, page, rowsPerPage]);

  const isAllSelected = filteredItems.length > 0 && filteredItems.every(item => selectedIds.has(item.id));
  const isSomeSelected = filteredItems.some(item => selectedIds.has(item.id)) && !isAllSelected;

  // Handlers
  const handleSelectAll = useCallback(() => {
    if (isAllSelected) {
      setSelectedIds(new Set());
    } else {
      setSelectedIds(new Set(filteredItems.map(item => item.id)));
    }
  }, [isAllSelected, filteredItems]);

  const handleSelectItem = useCallback((itemId: string) => {
    setSelectedIds(prev => {
      const newSet = new Set(prev);
      if (newSet.has(itemId)) {
        newSet.delete(itemId);
      } else {
        newSet.add(itemId);
      }
      return newSet;
    });
  }, []);

  const handleSelectRange = useCallback((startId: string, endId: string, _event: React.MouseEvent) => {
    if (!event.shiftKey) return;

    const startIndex = filteredItems.findIndex(item => item.id === startId);
    const endIndex = filteredItems.findIndex(item => item.id === endId);
    
    if (startIndex !== -1 && endIndex !== -1) {
      const range = filteredItems.slice(
        Math.min(startIndex, endIndex),
        Math.max(startIndex, endIndex) + 1
      );
      
      setSelectedIds(prev => {
        const newSet = new Set(prev);
        range.forEach(item => newSet.add(item.id));
        return newSet;
      });
    }
  }, [filteredItems]);

  const handleOperation = useCallback(async (operation: BulkOperation) => {
    if (selectedIds.size === 0) {
      setSnackbar({
        open: true,
        _message: 'No items selected',
        severity: 'warning'
      });
      return;
    }

    if (operation.requiresConfirmation) {
      setConfirmDialog({
        open: true,
        operation,
        _message: `Are you sure you want to ${operation.name.toLowerCase()} ${selectedIds.size} item(s)?`
      });
      return;
    }

    executeOperation(operation);
  }, [selectedIds]);

  const executeOperation = useCallback(async (operation: BulkOperation) => {
    setProgressDialog(true);
    setOperationProgress({
      operationId: operation.id,
      totalItems: selectedIds.size,
      processedItems: 0,
      failedItems: 0,
      status: 'processing',
      startTime: new Date()
    });

    // Simulate operation processing
    const selectedArray = Array.from(selectedIds);
    for (let i = 0; i < selectedArray.length; i++) {
      await new Promise(resolve => setTimeout(resolve, 100)); // Simulate processing time
      
      setOperationProgress(prev => ({
        ...prev!,
        processedItems: i + 1
      }));
    }

    // Complete operation
    setOperationProgress(prev => ({
      ...prev!,
      status: 'completed',
      endTime: new Date()
    }));

    // Add to history
    setHistory(prev => [...prev, {
      action: operation.name,
      items: selectedArray,
      timestamp: new Date()
    }]);
    setHistoryIndex(prev => prev + 1);

    // Callback
    if (onOperationComplete) {
      onOperationComplete(operation.id, selectedArray);
    }

    // Show success message
    setSnackbar({
      open: true,
      _message: `Successfully ${operation.name.toLowerCase()}d ${selectedIds.size} item(s)`,
      severity: 'success'
    });

    // Clear selection
    setSelectedIds(new Set());
    
    // Close dialogs
    setTimeout(() => {
      setProgressDialog(false);
      setOperationProgress(null);
    }, 1500);
  }, [selectedIds, onOperationComplete]);

  const handleUndo = useCallback(() => {
    if (historyIndex > 0) {
      const previousAction = history[historyIndex - 1];
      // Implement undo logic based on action
      setHistoryIndex(prev => prev - 1);
      setSnackbar({
        open: true,
        _message: `Undid: ${previousAction.action}`,
        severity: 'info'
      });
    }
  }, [history, historyIndex]);

  const handleRedo = useCallback(() => {
    if (historyIndex < history.length - 1) {
      const nextAction = history[historyIndex + 1];
      // Implement redo logic based on action
      setHistoryIndex(prev => prev + 1);
      setSnackbar({
        open: true,
        _message: `Redid: ${nextAction.action}`,
        severity: 'info'
      });
    }
  }, [history, historyIndex]);

  // Effects
  useEffect(() => {
    if (onSelectionChange) {
      onSelectionChange(Array.from(selectedIds));
    }
  }, [selectedIds, onSelectionChange]);

  useEffect(() => {
    setItems(initialItems);
  }, [initialItems]);

  // Render helpers
  const renderSelectionBar = () => (
    <Fade in={selectedIds.size > 0}>
      <Paper
        sx={{
          position: 'sticky',
          top: 0,
          zIndex: 10,
          p: 2,
          mb: 2,
          backgroundColor: alpha(theme.palette.primary.main, 0.1),
          borderRadius: 2,
          display: 'flex',
          alignItems: 'center',
          gap: 2
        }}
      >
        <Typography variant="subtitle1" sx={{ fontWeight: 600 }}>
          {selectedIds.size} item{selectedIds.size !== 1 ? 's' : ''} selected
        </Typography>
        
        <ButtonGroup size="small" variant="outlined">
          {operations.map(op => (
            <Tooltip key={op.id} title={op.name}>
              <Button
                onClick={() => handleOperation(op)}
                color={op.color}
                startIcon={op.icon}
              >
                {op.name}
              </Button>
            </Tooltip>
          ))}
        </ButtonGroup>

        <Box sx={{ flexGrow: 1 }} />

        <IconButton onClick={() => setSelectedIds(new Set())} size="small">
          <ClearIcon />
        </IconButton>
      </Paper>
    </Fade>
  );

  const renderTableView = () => (
    <TableContainer component={Paper} sx={{ maxHeight: 600 }}>
      <Table stickyHeader>
        <TableHead>
          <TableRow>
            <TableCell padding="checkbox">
              <Checkbox
                indeterminate={isSomeSelected}
                checked={isAllSelected}
                onChange={handleSelectAll}
              />
            </TableCell>
            <TableCell>Name</TableCell>
            <TableCell>Type</TableCell>
            <TableCell>Status</TableCell>
            <TableCell>Tags</TableCell>
            <TableCell>Modified</TableCell>
            <TableCell align="right">Actions</TableCell>
          </TableRow>
        </TableHead>
        <TableBody>
          {paginatedItems.map((item, index) => (
            <TableRow
              key={item.id}
              hover
              selected={selectedIds.has(item.id)}
              onClick={(_e) => handleSelectRange(
                index > 0 ? paginatedItems[index - 1].id : item.id,
                item.id,
                _e
              )}
            >
              <TableCell padding="checkbox">
                <Checkbox
                  checked={selectedIds.has(item.id)}
                  onChange={() => handleSelectItem(item.id)}
                  onClick={(_e) => _e.stopPropagation()}
                />
              </TableCell>
              <TableCell>
                <Box sx={{ display: 'flex', alignItems: 'center', gap: 1 }}>
                  {item.thumbnail && (
                    <Avatar src={item.thumbnail} sx={{ width: 32, height: 32 }}>
                      {getItemIcon(item.type)}
                    </Avatar>
                  )}
                  <Typography variant="body2">{item.name}</Typography>
                  {item.starred && <StarIcon fontSize="small" color="warning" />}
                </Box>
              </TableCell>
              <TableCell>
                <Chip label={item.type} size="small" variant="outlined" />
              </TableCell>
              <TableCell>
                <Chip
                  label={item.status}
                  size="small"
                  color={getStatusColor(item.status)}
                  variant="filled"
                />
              </TableCell>
              <TableCell>
                <Box sx={{ display: 'flex', gap: 0.5, flexWrap: 'wrap' }}>
                  {item.tags?.slice(0, 3).map(tag => (
                    <Chip key={tag} label={tag} size="small" />
                  ))}
                  {item.tags && item.tags.length > 3 && (
                    <Chip label={`+${item.tags.length - 3}`} size="small" variant="outlined" />
                  )}
                </Box>
              </TableCell>
              <TableCell>
                <Typography variant="caption">
                  {item.modifiedAt.toLocaleDateString()}
                </Typography>
              </TableCell>
              <TableCell align="right">
                <IconButton size="small" onClick={(_e) => {
                  _e.stopPropagation();
                  // Action menu would be implemented here
                }}>
                  <MoreVertIcon />
                </IconButton>
              </TableCell>
            </TableRow>
          ))}
        </TableBody>
      </Table>
      <TablePagination
        component="div"
        count={filteredItems.length}
        page={page}
        onPageChange={(e, newPage) => setPage(newPage)}
        rowsPerPage={rowsPerPage}
        onRowsPerPageChange={(_e) => {
          setRowsPerPage(parseInt(_e.target.value, 10));
          setPage(0);
        }}
      />
    </TableContainer>
  );

  const renderGridView = () => (
    <Grid container spacing={2}>
      {paginatedItems.map(item => (
        <Grid key={item.id} size={{ xs: 12, sm: 6, md: 4, lg: 3 }}>
          <Card
            sx={{
              position: 'relative',
              cursor: 'pointer',
              transition: 'all 0.2s',
              '&:hover': {
                transform: 'translateY(-2px)',
                boxShadow: 4
              },
              ...(selectedIds.has(item.id) && {
                borderColor: 'primary.main',
                borderWidth: 2,
                borderStyle: 'solid'
              })
            }}
            onClick={() => handleSelectItem(item.id)}
          >
            <Box sx={{ position: 'absolute', top: 8, left: 8, zIndex: 1 }}>
              <Checkbox
                checked={selectedIds.has(item.id)}
                onChange={() => handleSelectItem(item.id)}
                onClick={(_e) => _e.stopPropagation()}
                sx={{
                  backgroundColor: 'rgba(255, 255, 255, 0.9)',
                  borderRadius: 1
                }}
              />
            </Box>
            {item.starred && (
              <Box sx={{ position: 'absolute', top: 8, right: 8, zIndex: 1 }}>
                <StarIcon color="warning" />
              </Box>
            )}
            <CardContent>
              <Box sx={{ display: 'flex', alignItems: 'center', mb: 2 }}>
                {item.thumbnail ? (
                  <Avatar src={item.thumbnail} sx={{ width: 48, height: 48, mr: 2 }}>
                    {getItemIcon(item.type)}
                  </Avatar>
                ) : (
                  <Avatar sx={{ width: 48, height: 48, mr: 2 }}>
                    {getItemIcon(item.type)}
                  </Avatar>
                )}
                <Box sx={{ flexGrow: 1 }}>
                  <Typography variant="subtitle2" noWrap>
                    {item.name}
                  </Typography>
                  <Typography variant="caption" color="text.secondary">
                    {item.type}
                  </Typography>
                </Box>
              </Box>
              <Box sx={{ display: 'flex', gap: 0.5, mb: 1 }}>
                <Chip
                  label={item.status}
                  size="small"
                  color={getStatusColor(item.status)}
                />
              </Box>
              {item.tags && item.tags.length > 0 && (
                <Box sx={{ display: 'flex', gap: 0.5, flexWrap: 'wrap' }}>
                  {item.tags.slice(0, 2).map(tag => (
                    <Chip key={tag} label={tag} size="small" variant="outlined" />
                  ))}
                </Box>
              )}
            </CardContent>
          </Card>
        </Grid>
      ))}
    </Grid>
  );

  const getItemIcon = (type: BulkItem['type']) => {
    switch (type) {
      case 'channel': return <FolderIcon />;
      case 'video': return <VideoIcon />;
      case 'image': return <ImageIcon />;
      case 'file': return <FileIcon />;
      default: return <FileIcon />;
    }
  };

  const getStatusColor = (status: BulkItem['status']): 'default' | 'primary' | 'secondary' | 'error' | 'info' | 'success' | 'warning' => {
    switch (status) {
      case 'active': return 'success';
      case 'paused': return 'warning';
      case 'archived': return 'default';
      case 'processing': return 'info';
      default: return 'default';
    }
  };

  return (
    <Box>
      {/* Header Toolbar */}
      <Paper sx={{ p: 2, mb: 2 }}>
        <Grid container spacing={2} alignItems="center">
          <Grid size={{ xs: 12, sm: 6, md: 4 }}>
            <TextField
              fullWidth
              size="small"
              placeholder="Search items..."
              value={searchQuery}
              onChange={(_e) => setSearchQuery(_e.target.value)}
              InputProps={{
                startAdornment: (
                  <InputAdornment position="start">
                    <SearchIcon />
                  </InputAdornment>
                )
              }}
            />
          </Grid>
          
          <Grid size={{ xs: 6, sm: 3, md: 2 }}>
            <FormControl fullWidth size="small">
              <InputLabel>Type</InputLabel>
              <Select
                value={filterType}
                onChange={(_e) => setFilterType(_e.target.value as any)}
                label="Type"
              >
                <MenuItem value="all">All Types</MenuItem>
                <MenuItem value="channel">Channels</MenuItem>
                <MenuItem value="video">Videos</MenuItem>
                <MenuItem value="image">Images</MenuItem>
                <MenuItem value="file">Files</MenuItem>
              </Select>
            </FormControl>
          </Grid>
          
          <Grid size={{ xs: 6, sm: 3, md: 2 }}>
            <FormControl fullWidth size="small">
              <InputLabel>Sort By</InputLabel>
              <Select
                value={sortBy}
                onChange={(_e) => setSortBy(_e.target.value as any)}
                label="Sort By"
              >
                <MenuItem value="name">Name</MenuItem>
                <MenuItem value="date">Date Modified</MenuItem>
                <MenuItem value="type">Type</MenuItem>
              </Select>
            </FormControl>
          </Grid>
          
          <Grid size={{ xs: 12, sm: 12, md: 4 }}>
            <Box sx={{ display: 'flex', gap: 1, justifyContent: 'flex-end' }}>
            <ToggleButtonGroup
              value={viewMode}
              exclusive
              onChange={(e, value) => value && setViewMode(value)}
              size="small"
            >
              <ToggleButton value="table">
                <Tooltip title="Table View">
                  <TableIcon />
                </Tooltip>
              </ToggleButton>
              <ToggleButton value="grid">
                <Tooltip title="Grid View">
                  <GridIcon />
                </Tooltip>
              </ToggleButton>
            </ToggleButtonGroup>
            
            <ButtonGroup size="small">
              <Button
                onClick={handleUndo}
                disabled={historyIndex <= 0}
                startIcon={<UndoIcon />}
              >
                Undo
              </Button>
              <Button
                onClick={handleRedo}
                disabled={historyIndex >= history.length - 1}
                startIcon={<RedoIcon />}
              >
                Redo
              </Button>
            </ButtonGroup>
            </Box>
          </Grid>
        </Grid>
      </Paper>

      {/* Selection Bar */}
      {renderSelectionBar()}

      {/* Content Area */}
      {viewMode === 'table' ? renderTableView() : renderGridView()}

      {/* Confirmation Dialog */}
      <Dialog open={confirmDialog.open} onClose={() => setConfirmDialog({ open: false })}>
        <DialogTitle>Confirm Operation</DialogTitle>
        <DialogContent>
          <Typography>{confirmDialog.message}</Typography>
        </DialogContent>
        <DialogActions>
          <Button onClick={() => setConfirmDialog({ open: false })}>
            Cancel
          </Button>
          <Button
            onClick={() => {
              if (confirmDialog.operation) {
                executeOperation(confirmDialog.operation);
              }
              setConfirmDialog({ open: false });
            }}
            variant="contained"
            color={confirmDialog.operation?.color || 'primary'}
          >
            Confirm
          </Button>
        </DialogActions>
      </Dialog>

      {/* Progress Dialog */}
      <Dialog open={progressDialog} maxWidth="sm" fullWidth>
        <DialogTitle>Processing Operation</DialogTitle>
        <DialogContent>
          {operationProgress && (
            <Box>
              <Typography variant="body2" gutterBottom>
                Processing {operationProgress.processedItems} of {operationProgress.totalItems} items
              </Typography>
              <LinearProgress
                variant="determinate"
                value={(operationProgress.processedItems / operationProgress.totalItems) * 100}
                sx={{ mb: 2 }}
              />
              {operationProgress.status === 'completed' && (
                <Alert severity="success">
                  Operation completed successfully!
                </Alert>
              )}
            </Box>
          )}
        </DialogContent>
      </Dialog>

      {/* Snackbar */}
      <Snackbar
        open={snackbar.open}
        autoHideDuration={4000}
        onClose={() => setSnackbar(prev => ({ ...prev, open: false }))}
        anchorOrigin={{ vertical: 'bottom', horizontal: 'center' }}
      >
        <Alert
          onClose={() => setSnackbar(prev => ({ ...prev, open: false }))}
          severity={snackbar.severity}
          variant="filled"
        >
          {snackbar.message}
        </Alert>
      </Snackbar>

      {/* Speed Dial for Quick Actions */}
      <SpeedDial
        ariaLabel="Quick Actions"
        sx={{ position: 'fixed', bottom: 16, right: 16 }}
        icon={<SpeedDialIcon />}
        open={speedDialOpen}
        onOpen={() => setSpeedDialOpen(true)}
        onClose={() => setSpeedDialOpen(false)}
      >
        <SpeedDialAction
          icon={<SelectAllIcon />}
          tooltipTitle="Select All"
          onClick={() => {
            handleSelectAll();
            setSpeedDialOpen(false);
          }}
        />
        <SpeedDialAction
          icon={<DeselectAllIcon />}
          tooltipTitle="Clear Selection"
          onClick={() => {
            setSelectedIds(new Set());
            setSpeedDialOpen(false);
          }}
        />
        <SpeedDialAction
          icon={<RefreshIcon />}
          tooltipTitle="Refresh"
          onClick={() => {
            // Refresh logic
            setSpeedDialOpen(false);
          }}
        />
      </SpeedDial>
    </Box>
  );
};