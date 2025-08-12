import React, { useState, useCallback, useMemo } from 'react';
import {
  Box,
  Card,
  CardContent,
  CardHeader,
  IconButton,
  Menu,
  MenuItem,
  Typography,
  Chip,
  Dialog,
  DialogTitle,
  DialogContent,
  DialogActions,
  Button,
  Grid,
  FormControl,
  InputLabel,
  Select,
  TextField,
  Switch,
  FormControlLabel,
  Divider,
  List,
  ListItem,
  ListItemIcon,
  ListItemText,
  ListItemSecondaryAction,
  Checkbox,
  Tooltip,
  Paper,
  Fab,
  Zoom,
} from '@mui/material';
import {
  DragIndicator,
  Close,
  Settings,
  Add,
  MoreVert,
  Fullscreen,
  FullscreenExit,
  Refresh,
  Download,
  Visibility,
  VisibilityOff,
  Edit,
  Delete,
  ContentCopy,
  Lock,
  LockOpen,
  TrendingUp,
  AttachMoney,
  VideoLibrary,
  Analytics,
  Speed,
  Warning,
  CheckCircle,
  CloudQueue,
  Schedule,
} from '@mui/icons-material';
import { DragDropContext, Droppable, Draggable } from 'react-beautiful-dnd';
import { format } from 'date-fns';

// Widget Types
export interface Widget {
  id: string;
  type: 'metric' | 'chart' | 'list' | 'progress' | 'custom';
  title: string;
  size: 'small' | 'medium' | 'large' | 'full';
  position: { x: number; y: number };
  config: any;
  locked?: boolean;
  visible?: boolean;
  refreshInterval?: number;
  lastUpdated?: Date;
}

interface WidgetLibraryItem {
  id: string;
  type: Widget['type'];
  title: string;
  description: string;
  icon: React.ReactNode;
  defaultConfig: any;
  sizes: Widget['size'][];
}

const widgetLibrary: WidgetLibraryItem[] = [
  {
    id: 'revenue-metric',
    type: 'metric',
    title: 'Revenue Tracker',
    description: 'Track daily, weekly, and monthly revenue',
    icon: <AttachMoney />,
    defaultConfig: { metric: 'revenue', period: 'daily' },
    sizes: ['small', 'medium'],
  },
  {
    id: 'video-performance',
    type: 'chart',
    title: 'Video Performance',
    description: 'View performance metrics for your videos',
    icon: <VideoLibrary />,
    defaultConfig: { chartType: 'line', metrics: ['views', 'engagement'] },
    sizes: ['medium', 'large', 'full'],
  },
  {
    id: 'processing-queue',
    type: 'list',
    title: 'Processing Queue',
    description: 'Monitor videos currently being processed',
    icon: <CloudQueue />,
    defaultConfig: { maxItems: 5, showStatus: true },
    sizes: ['medium', 'large'],
  },
  {
    id: 'channel-health',
    type: 'progress',
    title: 'Channel Health',
    description: 'Overall health score of your channels',
    icon: <Speed />,
    defaultConfig: { showBreakdown: true },
    sizes: ['small', 'medium'],
  },
  {
    id: 'trend-analysis',
    type: 'chart',
    title: 'Trend Analysis',
    description: 'Analyze trending topics and niches',
    icon: <TrendingUp />,
    defaultConfig: { chartType: 'heatmap', period: '7d' },
    sizes: ['large', 'full'],
  },
  {
    id: 'scheduled-uploads',
    type: 'list',
    title: 'Scheduled Uploads',
    description: 'View upcoming scheduled video uploads',
    icon: <Schedule />,
    defaultConfig: { maxItems: 10, groupByDay: true },
    sizes: ['medium', 'large'],
  },
];

interface CustomizableWidgetsProps {
  initialWidgets?: Widget[];
  onSave?: (widgets: Widget[]) => void;
  allowEdit?: boolean;
}

export const CustomizableWidgets: React.FC<CustomizableWidgetsProps> = ({
  initialWidgets = [],
  onSave,
  allowEdit = true,
}) => {
  const [widgets, setWidgets] = useState<Widget[]>(initialWidgets);
  const [editMode, setEditMode] = useState(false);
  const [addDialogOpen, setAddDialogOpen] = useState(false);
  const [configDialogOpen, setConfigDialogOpen] = useState(false);
  const [selectedWidget, setSelectedWidget] = useState<Widget | null>(null);
  const [fullscreenWidget, setFullscreenWidget] = useState<string | null>(null);
  const [anchorEl, setAnchorEl] = useState<null | HTMLElement>(null);
  const [menuWidget, setMenuWidget] = useState<string | null>(null);

  // Handle drag and drop
  const handleDragEnd = (result: any) => {
    if (!result.destination) return;

    const items = Array.from(widgets);
    const [reorderedItem] = items.splice(result.source.index, 1);
    items.splice(result.destination.index, 0, reorderedItem);

    setWidgets(items);
  };

  // Add new widget
  const handleAddWidget = (libraryItem: WidgetLibraryItem, size: Widget['size']) => {
    const newWidget: Widget = {
      id: `widget-${Date.now()}`,
      type: libraryItem.type,
      title: libraryItem.title,
      size,
      position: { x: 0, y: widgets.length },
      config: libraryItem.defaultConfig,
      visible: true,
      lastUpdated: new Date(),
    };

    setWidgets([...widgets, newWidget]);
    setAddDialogOpen(false);
  };

  // Remove widget
  const handleRemoveWidget = (widgetId: string) => {
    setWidgets(widgets.filter(w => w.id !== widgetId));
    setAnchorEl(null);
  };

  // Toggle widget visibility
  const handleToggleVisibility = (widgetId: string) => {
    setWidgets(widgets.map(w =>
      w.id === widgetId ? { ...w, visible: !w.visible } : w
    ));
  };

  // Toggle widget lock
  const handleToggleLock = (widgetId: string) => {
    setWidgets(widgets.map(w =>
      w.id === widgetId ? { ...w, locked: !w.locked } : w
    ));
  };

  // Duplicate widget
  const handleDuplicateWidget = (widgetId: string) => {
    const widget = widgets.find(w => w.id === widgetId);
    if (widget) {
      const newWidget: Widget = {
        ...widget,
        id: `widget-${Date.now()}`,
        title: `${widget.title} (Copy)`,
      };
      setWidgets([...widgets, newWidget]);
    }
    setAnchorEl(null);
  };

  // Refresh widget data
  const handleRefreshWidget = (widgetId: string) => {
    setWidgets(widgets.map(w =>
      w.id === widgetId ? { ...w, lastUpdated: new Date() } : w
    ));
    // Trigger actual data refresh here
  };

  // Export widget data
  const handleExportWidget = (widgetId: string) => {
    const widget = widgets.find(w => w.id === widgetId);
    if (widget) {
      const data = JSON.stringify(widget, null, 2);
      const blob = new Blob([data], { type: 'application/json' });
      const url = URL.createObjectURL(blob);
      const a = document.createElement('a');
      a.href = url;
      a.download = `${widget.title.replace(/\s+/g, '-')}-${format(new Date(), 'yyyy-MM-dd')}.json`;
      a.click();
      URL.revokeObjectURL(url);
    }
    setAnchorEl(null);
  };

  // Widget menu actions
  const handleWidgetMenu = (event: React.MouseEvent<HTMLElement>, widgetId: string) => {
    setAnchorEl(event.currentTarget);
    setMenuWidget(widgetId);
  };

  const handleCloseMenu = () => {
    setAnchorEl(null);
    setMenuWidget(null);
  };

  // Save widgets configuration
  const handleSaveConfiguration = () => {
    onSave?.(widgets);
    setEditMode(false);
  };

  // Render individual widget
  const renderWidget = (widget: Widget) => {
    const gridSizes = {
      small: { xs: 12, sm: 6, md: 3 },
      medium: { xs: 12, sm: 12, md: 6 },
      large: { xs: 12, sm: 12, md: 9 },
      full: { xs: 12, sm: 12, md: 12 },
    };

    const size = gridSizes[widget.size];

    return (
      <Grid item {...size} key={widget.id}>
        <Card
          sx={{
            height: '100%',
            opacity: widget.visible ? 1 : 0.5,
            position: 'relative',
            ...(editMode && !widget.locked && {
              cursor: 'move',
              '&:hover': {
                boxShadow: 4,
              },
            }),
          }}
        >
          {editMode && !widget.locked && (
            <Box
              sx={{
                position: 'absolute',
                top: 8,
                left: 8,
                zIndex: 1,
              }}
            >
              <DragIndicator color="action" />
            </Box>
          )}

          <CardHeader
            title={
              <Box sx={{ display: 'flex', alignItems: 'center', gap: 1 }}>
                <Typography variant="h6" fontSize={16}>
                  {widget.title}
                </Typography>
                {widget.locked && (
                  <Tooltip title="Widget is locked">
                    <Lock fontSize="small" color="action" />
                  </Tooltip>
                )}
                {!widget.visible && (
                  <Tooltip title="Widget is hidden">
                    <VisibilityOff fontSize="small" color="action" />
                  </Tooltip>
                )}
              </Box>
            }
            action={
              <Box>
                {widget.lastUpdated && (
                  <Typography variant="caption" color="text.secondary" sx={{ mr: 1 }}>
                    {format(widget.lastUpdated, 'HH:mm')}
                  </Typography>
                )}
                <IconButton
                  size="small"
                  onClick={(e) => handleWidgetMenu(e, widget.id)}
                >
                  <MoreVert fontSize="small" />
                </IconButton>
              </Box>
            }
            sx={{ pb: 1 }}
          />

          <CardContent>
            {/* Widget content based on type */}
            {widget.type === 'metric' && (
              <Box>
                <Typography variant="h3" fontWeight="bold">
                  $1,234.56
                </Typography>
                <Box sx={{ display: 'flex', alignItems: 'center', gap: 0.5, mt: 1 }}>
                  <TrendingUp color="success" fontSize="small" />
                  <Typography variant="body2" color="success.main">
                    +12.5% from yesterday
                  </Typography>
                </Box>
              </Box>
            )}

            {widget.type === 'progress' && (
              <Box>
                <Box sx={{ display: 'flex', justifyContent: 'space-between', mb: 1 }}>
                  <Typography variant="body2">Channel Health</Typography>
                  <Typography variant="body2" fontWeight="bold">85%</Typography>
                </Box>
                <Box sx={{ width: '100%', bgcolor: 'grey.200', borderRadius: 1, height: 8 }}>
                  <Box
                    sx={{
                      width: '85%',
                      bgcolor: 'success.main',
                      borderRadius: 1,
                      height: '100%',
                    }}
                  />
                </Box>
              </Box>
            )}

            {widget.type === 'list' && (
              <List dense>
                {[1, 2, 3].map((item) => (
                  <ListItem key={item}>
                    <ListItemIcon>
                      <CloudQueue color="primary" />
                    </ListItemIcon>
                    <ListItemText
                      primary={`Video ${item}`}
                      secondary="Processing..."
                    />
                    <ListItemSecondaryAction>
                      <Chip label="50%" size="small" />
                    </ListItemSecondaryAction>
                  </ListItem>
                ))}
              </List>
            )}

            {widget.type === 'chart' && (
              <Box sx={{ height: 200, display: 'flex', alignItems: 'center', justifyContent: 'center' }}>
                <Typography variant="body2" color="text.secondary">
                  Chart visualization here
                </Typography>
              </Box>
            )}
          </CardContent>
        </Card>
      </Grid>
    );
  };

  return (
    <Box>
      {/* Edit Mode Toggle */}
      {allowEdit && (
        <Box sx={{ mb: 2, display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
          <FormControlLabel
            control={
              <Switch
                checked={editMode}
                onChange={(e) => setEditMode(e.target.checked)}
              />
            }
            label="Edit Dashboard"
          />
          
          {editMode && (
            <Box sx={{ display: 'flex', gap: 1 }}>
              <Button
                variant="outlined"
                startIcon={<Add />}
                onClick={() => setAddDialogOpen(true)}
              >
                Add Widget
              </Button>
              <Button
                variant="contained"
                onClick={handleSaveConfiguration}
              >
                Save Layout
              </Button>
            </Box>
          )}
        </Box>
      )}

      {/* Widgets Grid */}
      {editMode ? (
        <DragDropContext onDragEnd={handleDragEnd}>
          <Droppable droppableId="widgets">
            {(provided) => (
              <Grid
                container
                spacing={2}
                {...provided.droppableProps}
                ref={provided.innerRef}
              >
                {widgets.map((widget, index) => (
                  <Draggable
                    key={widget.id}
                    draggableId={widget.id}
                    index={index}
                    isDragDisabled={widget.locked}
                  >
                    {(provided) => (
                      <div
                        ref={provided.innerRef}
                        {...provided.draggableProps}
                        {...provided.dragHandleProps}
                        style={{
                          ...provided.draggableProps.style,
                          width: '100%',
                        }}
                      >
                        {renderWidget(widget)}
                      </div>
                    )}
                  </Draggable>
                ))}
                {provided.placeholder}
              </Grid>
            )}
          </Droppable>
        </DragDropContext>
      ) : (
        <Grid container spacing={2}>
          {widgets.filter(w => w.visible).map(renderWidget)}
        </Grid>
      )}

      {/* Widget Menu */}
      <Menu
        anchorEl={anchorEl}
        open={Boolean(anchorEl)}
        onClose={handleCloseMenu}
      >
        <MenuItem onClick={() => {
          if (menuWidget) {
            setFullscreenWidget(menuWidget);
            handleCloseMenu();
          }
        }}>
          <ListItemIcon>
            <Fullscreen fontSize="small" />
          </ListItemIcon>
          <ListItemText>Fullscreen</ListItemText>
        </MenuItem>
        
        <MenuItem onClick={() => {
          if (menuWidget) {
            handleRefreshWidget(menuWidget);
            handleCloseMenu();
          }
        }}>
          <ListItemIcon>
            <Refresh fontSize="small" />
          </ListItemIcon>
          <ListItemText>Refresh</ListItemText>
        </MenuItem>
        
        <MenuItem onClick={() => {
          if (menuWidget) {
            setSelectedWidget(widgets.find(w => w.id === menuWidget) || null);
            setConfigDialogOpen(true);
            handleCloseMenu();
          }
        }}>
          <ListItemIcon>
            <Settings fontSize="small" />
          </ListItemIcon>
          <ListItemText>Configure</ListItemText>
        </MenuItem>
        
        <Divider />
        
        <MenuItem onClick={() => {
          if (menuWidget) {
            handleToggleVisibility(menuWidget);
            handleCloseMenu();
          }
        }}>
          <ListItemIcon>
            <VisibilityOff fontSize="small" />
          </ListItemIcon>
          <ListItemText>Hide</ListItemText>
        </MenuItem>
        
        <MenuItem onClick={() => {
          if (menuWidget) {
            handleToggleLock(menuWidget);
            handleCloseMenu();
          }
        }}>
          <ListItemIcon>
            <Lock fontSize="small" />
          </ListItemIcon>
          <ListItemText>Lock/Unlock</ListItemText>
        </MenuItem>
        
        <MenuItem onClick={() => {
          if (menuWidget) {
            handleDuplicateWidget(menuWidget);
          }
        }}>
          <ListItemIcon>
            <ContentCopy fontSize="small" />
          </ListItemIcon>
          <ListItemText>Duplicate</ListItemText>
        </MenuItem>
        
        <MenuItem onClick={() => {
          if (menuWidget) {
            handleExportWidget(menuWidget);
          }
        }}>
          <ListItemIcon>
            <Download fontSize="small" />
          </ListItemIcon>
          <ListItemText>Export</ListItemText>
        </MenuItem>
        
        <Divider />
        
        <MenuItem onClick={() => {
          if (menuWidget) {
            handleRemoveWidget(menuWidget);
          }
        }} sx={{ color: 'error.main' }}>
          <ListItemIcon>
            <Delete fontSize="small" color="error" />
          </ListItemIcon>
          <ListItemText>Remove</ListItemText>
        </MenuItem>
      </Menu>

      {/* Add Widget Dialog */}
      <Dialog
        open={addDialogOpen}
        onClose={() => setAddDialogOpen(false)}
        maxWidth="md"
        fullWidth
      >
        <DialogTitle>Add Widget</DialogTitle>
        <DialogContent>
          <Grid container spacing={2} sx={{ mt: 1 }}>
            {widgetLibrary.map((item) => (
              <Grid item xs={12} sm={6} key={item.id}>
                <Paper
                  sx={{
                    p: 2,
                    cursor: 'pointer',
                    '&:hover': { bgcolor: 'action.hover' },
                  }}
                >
                  <Box sx={{ display: 'flex', gap: 2 }}>
                    <Box sx={{ color: 'primary.main' }}>
                      {item.icon}
                    </Box>
                    <Box sx={{ flex: 1 }}>
                      <Typography variant="subtitle1" fontWeight="medium">
                        {item.title}
                      </Typography>
                      <Typography variant="caption" color="text.secondary">
                        {item.description}
                      </Typography>
                      <Box sx={{ mt: 1, display: 'flex', gap: 1 }}>
                        {item.sizes.map((size) => (
                          <Button
                            key={size}
                            size="small"
                            variant="outlined"
                            onClick={() => handleAddWidget(item, size)}
                          >
                            {size}
                          </Button>
                        ))}
                      </Box>
                    </Box>
                  </Box>
                </Paper>
              </Grid>
            ))}
          </Grid>
        </DialogContent>
        <DialogActions>
          <Button onClick={() => setAddDialogOpen(false)}>Cancel</Button>
        </DialogActions>
      </Dialog>

      {/* Floating Action Button for mobile */}
      {allowEdit && !editMode && (
        <Zoom in>
          <Fab
            color="primary"
            sx={{
              position: 'fixed',
              bottom: 16,
              right: 16,
            }}
            onClick={() => setEditMode(true)}
          >
            <Edit />
          </Fab>
        </Zoom>
      )}
    </Box>
  );
};