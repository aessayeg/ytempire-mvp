/**
 * Video Queue Interface Component
 * Comprehensive interface for managing video generation queue
 */

import React, { useState, useEffect, useMemo, useCallback } from 'react';
import { 
  DragDropContext,
  Droppable,
  Draggable,
  DropResult
 } from '@hello-pangea/dnd';
import { 
  Box,
  Paper,
  Typography,
  Button,
  IconButton,
  Chip,
  LinearProgress,
  Menu,
  MenuItem,
  Select,
  FormControl,
  InputLabel,
  Grid,
  Card,
  CardContent,
  Badge,
  Alert,
  Skeleton
 } from '@mui/material';
import { 
  PlayArrow,
  Pause,
  Delete,
  Edit,
  MoreVert,
  Schedule,
  CheckCircle,
  Error,
  Info,
  Sort,
  Refresh,
  Add,
  DragIndicator,
  Timeline,
  Speed,
  AttachMoney
 } from '@mui/icons-material';
import {  format, formatDistanceToNow, addMinutes  } from 'date-fns';
import {  useOptimizedStore  } from '../../stores/optimizedStore';
import {  api  } from '../../services/api';

interface VideoQueueItem {
  id: string;
  channelId: string;
  title: string;
  description?: string;
  topic: string;
  style: string;
  duration: number;
  status: 'pending' | 'scheduled' | 'processing' | 'completed' | 'failed' | 'paused';
  priority: 'low' | 'normal' | 'high' | 'urgent';
  progress: number;
  scheduledTime?: string;
  estimatedCost: number;
  processingTime: number;
  error?: string;
  retryCount: number;
  metadata: {
    thumbnailStyle?: string;
    voiceStyle?: string;
    targetAudience?: string;
    keywords?: string[];
    autoPublish?: boolean;
  };
}

interface QueueStats {
  totalItems: number;
  pending: number;
  processing: number;
  completed: number;
  failed: number;
  estimatedTotalCost: number;
  estimatedCompletionTime?: string;
  processingRate: number;
}

interface TabPanelProps {
  children?: React.ReactNode;
  index: number;
  value: number;
}

function TabPanel(props: TabPanelProps) {
  const { children, value, index, ...other } = props;
  return (
    <div
      role="tabpanel"
      hidden={value !== index}
      id={`queue-tabpanel-${index}`}
      aria-labelledby={`queue-tab-${index}`}
      {...other}
    >
      {value === index && <Box sx={{ p: 3 }}>{children}</Box>}
    </div>
  );
}

export const VideoQueueInterface: React.FC = () => {
  const [queueItems, setQueueItems] = useState<VideoQueueItem[]>([]);
  const [stats, setStats] = useState<QueueStats | null>(null);
  const [loading, setLoading] = useState(true);
  const [selectedTab, setSelectedTab] = useState(0);
  const [filterStatus, setFilterStatus] = useState<string>('all');
  const [sortBy, setSortBy] = useState<string>('priority');
  const [selectedItem, setSelectedItem] = useState<VideoQueueItem | null>(null);
  const [editDialogOpen, setEditDialogOpen] = useState(false);
  const [anchorEl, setAnchorEl] = useState<null | HTMLElement>(null);
  const [refreshing, setRefreshing] = useState(false);

  const { channels, addNotification } = useOptimizedStore();

  // Fetch queue items
  const fetchQueue = useCallback(async () => {
    try {
      setRefreshing(true);
      const response = await api.get('/queue/list');
      setQueueItems(response.data);
      
      // Fetch stats
      const statsResponse = await api.get('/queue/stats/summary');
      setStats(statsResponse.data);
      
      setLoading(false);
    } catch (error) {
      console.error('Failed to fetch queue:', error);
      addNotification({
        type: 'error',
        message: 'Failed to load video queue'
      });
      setLoading(false);
    } finally {
      setRefreshing(false);
    }
  }, [addNotification]);

  useEffect(() => {
    fetchQueue();
    // Refresh every 30 seconds
    const interval = setInterval(fetchQueue, 30000);
    return () => clearInterval(interval);
  }, [fetchQueue]);

  // Filter and sort queue items
  const filteredItems = useMemo(() => {
    let filtered = [...queueItems];

    // Apply status filter
    if (filterStatus !== 'all') {
      filtered = filtered.filter((item) => item.status === filterStatus);
    }

    // Apply sorting
    filtered.sort((a, b) => {
      switch (sortBy) {
        case 'priority':
          const priorityOrder = { urgent: 0, high: 1, normal: 2, low: 3 };
          return priorityOrder[a.priority] - priorityOrder[b.priority];
        case 'scheduled':
          return (a.scheduledTime || '').localeCompare(b.scheduledTime || '');
        case 'cost':
          return b.estimatedCost - a.estimatedCost;
        case 'duration':
          return b.duration - a.duration;
        default:
          return 0;
      }
    });

    return filtered;
  }, [queueItems, filterStatus, sortBy]);

  // Group items by status for tabs
  const itemsByStatus = useMemo(() => {
    return {
      all: queueItems,
      pending: queueItems.filter((item) => item.status === 'pending'),
      processing: queueItems.filter((item) => item.status === 'processing'),
      completed: queueItems.filter((item) => item.status === 'completed'),
      failed: queueItems.filter((item) => item.status === 'failed')
    };
  }, [queueItems]);

  // Handle drag and drop
  const handleDragEnd = async (result: DropResult) => {
    if (!result.destination) return;

    const items = Array.from(filteredItems);
    const [reorderedItem] = items.splice(result.source.index, 1);
    items.splice(result.destination.index, 0, reorderedItem);

    // Update priority based on new position
    const updatedItems = items.map((item, index) => ({
      ...item,
      priority: index === 0 ? 'urgent' : index < 3 ? 'high' : 'normal'
    }));

    setQueueItems(updatedItems);

    // Update on server
    try {
      await api.patch(`/queue/${reorderedItem.id}`, {
        priority: updatedItems.find((i) => i.id === reorderedItem.id)?.priority
      });
    } catch (error) {
      console.error('Failed to update priority:', error);
    }
  };
  // Handle actions
  const handlePause = async (id: string) => {
    try {
      await api.patch(`/queue/${id}`, { status: 'paused' });
      fetchQueue();
      addNotification({
        type: 'success',
        message: 'Video paused'
      });
    } catch (error) {
      addNotification({
        type: 'error',
        message: 'Failed to pause video'
      });
    }
  };
  const handleResume = async (id: string) => {
    try {
      await api.patch(`/queue/${id}`, { status: 'pending' });
      fetchQueue();
      addNotification({
        type: 'success',
        message: 'Video resumed'
      });
    } catch (error) {
      addNotification({
        type: 'error',
        message: 'Failed to resume video'
      });
    }
  };
  const handleRetry = async (id: string) => {
    try {
      await api.post(`/queue/${id}/retry`);
      fetchQueue();
      addNotification({
        type: 'success',
        message: 'Video queued for retry'
      });
    } catch (error) {
      addNotification({
        type: 'error',
        message: 'Failed to retry video'
      });
    }
  };
  const handleDelete = async (id: string) => {
    try {
      await api.delete(`/queue/${id}`);
      fetchQueue();
      addNotification({
        type: 'success',
        message: 'Video removed from queue'
      });
    } catch (error) {
      addNotification({
        type: 'error',
        message: 'Failed to remove video'
      });
    }
  };
  const handlePauseAll = async () => {
    try {
      await api.post('/queue/pause-all');
      fetchQueue();
      addNotification({
        type: 'success',
        message: 'All videos paused'
      });
    } catch (error) {
      addNotification({
        type: 'error',
        message: 'Failed to pause all videos'
      });
    }
  };
  const handleResumeAll = async () => {
    try {
      await api.post('/queue/resume-all');
      fetchQueue();
      addNotification({
        type: 'success',
        message: 'All videos resumed'
      });
    } catch (error) {
      addNotification({
        type: 'error',
        message: 'Failed to resume all videos'
      });
    }
  };
  // Render queue item
  const renderQueueItem = (item: VideoQueueItem, _index: number) => {
    const getStatusIcon = () => {
      switch (item.status) {
        case 'processing':
          return <Speed color="primary" />;
        case 'completed':
          return <CheckCircle color="success" />;
        case 'failed':
          return <Error color="error" />;
        case 'scheduled':
          return <Schedule color="action" />;
        case 'paused':
          return <Pause color="warning" />;
        default:
          return <Info color="info" />;
      }
    };
    const getPriorityColor = () => {
      switch (item.priority) {
        case 'urgent':
          return 'error';
        case 'high':
          return 'warning';
        case 'normal':
          return 'info';
        case 'low':
          return 'default';
        default:
          return 'default';
      }
    };
    const channel = channels.list.find((c) => c.id === item.channelId);

    return (
    <Draggable key={item.id} draggableId={item.id} index={index}>
        {(provided, snapshot) => (
          <Card
            ref={provided.innerRef}
            {...provided.draggableProps}
            sx={ {
              mb: 2,
              opacity: snapshot.isDragging ? 0.8 : 1,
              transform: snapshot.isDragging ? 'rotate(2deg)' : 'none',
              transition: 'all 0.2s ease',
              '&:hover': {
                boxShadow: 3
              }
            }}
          >
            <CardContent>
              <Grid container spacing={2} alignItems="center">
                <Grid item {...provided.dragHandleProps}>
                  <DragIndicator color="action" />
                </Grid>
      <Grid item>
                  {getStatusIcon()}
                </Grid>
                <Grid item xs>
                  <Typography variant="h6" gutterBottom>
                    {item.title}
                  </Typography>
                  <Box display="flex" gap={1} flexWrap="wrap">
                    <Chip
                      size="small"
                      label={channel?.name || 'Unknown Channel'}
                      variant="outlined"
                    />
                    <Chip
                      size="small"
                      label={item.priority}
                      color={getPriorityColor()}
                    />
                    <Chip
                      size="small"
                      icon={<AttachMoney />}
                      label={`$${item.estimatedCost.toFixed(2)}`}
                      variant="outlined"
                    />
                    <Chip
                      size="small"
                      icon={<Timeline />}
                      label={`${item.duration} min`}
                      variant="outlined"
                    />
                    {item.scheduledTime && (
                      <Chip
                        size="small"
                        icon={<Schedule />}
                        label={format(new Date(item.scheduledTime), 'MMM d, h:mm a')}
                        variant="outlined"
                      />
                    )}
                  </Box>
                  {item.description && (
                    <Typography variant="body2" color="text.secondary" sx={{ mt: 1 }}>
                      {item.description}
                    </Typography>
                  )}
                  {item.status === 'processing' && (
                    <Box sx={{ mt: 2 }}>
                      <Box display="flex" justifyContent="space-between">
                        <Typography variant="caption">
                          Processing: {item.progress}%
                        </Typography>
                        <Typography variant="caption">
                          Est. {formatDistanceToNow(
                            addMinutes(new Date(), item.processingTime - item.progress / 100 * item.processingTime)
                          )}
                        </Typography>
                      </Box>
                      <LinearProgress
                        variant="determinate"
                        value={item.progress}
                        sx={{ mt: 1 }}
                      />
                    </Box>
                  )}
                  {item.status === 'failed' && item.error && (
                    <Alert severity="error" sx={{ mt: 1 }}>
                      {item.error} (Retry {item.retryCount}/3)
                    </Alert>
                  )}
                </Grid>
                <Grid item>
                  <IconButton
                    onClick={(e) => {
                      setAnchorEl(e.currentTarget);
                      setSelectedItem(item);
                    }}
                  >
                    <MoreVert />
                  </IconButton>
                </Grid>
              </Grid>
            </CardContent>
          </Card>
        )}
      </Draggable>
    );
  };
  return (
    <Box>
      {/* Header */}
      <Paper sx={{ p: 2, mb: 3 }}>
        <Grid container spacing={2} alignItems="center">
          <Grid item xs>
            <Typography variant="h5">Video Generation Queue</Typography>
            {stats && (
              <Typography variant="body2" color="text.secondary">
                {stats.totalItems} total • {stats.processing} processing • 
                Est. cost: ${stats.estimatedTotalCost.toFixed(2)} • 
                Rate: {stats.processingRate.toFixed(1)} videos/hour
              </Typography>
            )}
          </Grid>
      <Grid item>
            <Button
              startIcon={<Add />}
              variant="contained"
              onClick={() => {/* Open add dialog */}}
            >
              Add Video
            </Button>
          </Grid>
          <Grid item>
            <Button
              startIcon={<Pause />}
              onClick={handlePauseAll}
            >
              Pause All
            </Button>
          </Grid>
          <Grid item>
            <Button
              startIcon={<PlayArrow />}
              onClick={handleResumeAll}
            >
              Resume All
            </Button>
          </Grid>
          <Grid item>
            <IconButton onClick={fetchQueue} disabled={refreshing}>
              <Badge badgeContent={refreshing ? '!' : 0} color="primary">
                <Refresh />
              </Badge>
            </IconButton>
          </Grid>
        </Grid>
      </Paper>

      {/* Stats Cards */}
      {stats && (
        <Grid container spacing={2} sx={{ mb: 3 }}>
          <Grid item xs={12} sm={6} md={3}>
            <Card>
              <CardContent>
                <Typography color="text.secondary" gutterBottom>
                  Pending
                </Typography>
                <Typography variant="h4">
                  {stats.pending}
                </Typography>
              </CardContent>
            </Card>
          </Grid>
          <Grid item xs={12} sm={6} md={3}>
            <Card>
              <CardContent>
                <Typography color="text.secondary" gutterBottom>
                  Processing
                </Typography>
                <Typography variant="h4" color="primary">
                  {stats.processing}
                </Typography>
              </CardContent>
            </Card>
          </Grid>
          <Grid item xs={12} sm={6} md={3}>
            <Card>
              <CardContent>
                <Typography color="text.secondary" gutterBottom>
                  Completed
                </Typography>
                <Typography variant="h4" color="success.main">
                  {stats.completed}
                </Typography>
              </CardContent>
            </Card>
          </Grid>
          <Grid item xs={12} sm={6} md={3}>
            <Card>
              <CardContent>
                <Typography color="text.secondary" gutterBottom>
                  Failed
                </Typography>
                <Typography variant="h4" color="error.main">
                  {stats.failed}
                </Typography>
              </CardContent>
            </Card>
          </Grid>
        </Grid>
      )}
      {/* Filters and Sorting */}
      <Paper sx={{ p: 2, mb: 3 }}>
        <Grid container spacing={2} alignItems="center">
          <Grid item>
            <FormControl size="small" sx={{ minWidth: 120 }}>
              <InputLabel>Status</InputLabel>
              <Select
                value={filterStatus}
                label="Status"
                onChange={(e) => setFilterStatus(e.target.value)}
              >
                <MenuItem value="all">All</MenuItem>
                <MenuItem value="pending">Pending</MenuItem>
                <MenuItem value="scheduled">Scheduled</MenuItem>
                <MenuItem value="processing">Processing</MenuItem>
                <MenuItem value="completed">Completed</MenuItem>
                <MenuItem value="failed">Failed</MenuItem>
                <MenuItem value="paused">Paused</MenuItem>
              </Select>
            </FormControl>
          </Grid>
          <Grid item>
            <FormControl size="small" sx={{ minWidth: 120 }}>
              <InputLabel>Sort By</InputLabel>
              <Select
                value={sortBy}
                label="Sort By"
                onChange={(e) => setSortBy(e.target.value)}
              >
                <MenuItem value="priority">Priority</MenuItem>
                <MenuItem value="scheduled">Scheduled Time</MenuItem>
                <MenuItem value="cost">Cost</MenuItem>
                <MenuItem value="duration">Duration</MenuItem>
              </Select>
            </FormControl>
          </Grid>
        </Grid>
      </Paper>

      {/* Queue Items */}
      {loading ? (
        <Box>
          {[1, 2, 3].map((i) => (
            <Skeleton key={i} variant="rectangular" height={120} sx={{ mb: 2 }} />
          ))}
        </Box>
      ) : (
        <DragDropContext onDragEnd={handleDragEnd}>
          <Droppable droppableId="queue">
            {(provided) => (
              <Box {...provided.droppableProps} ref={provided.innerRef}>
                {filteredItems.length === 0 ? (
                  <Paper sx={{ p: 4, textAlign: 'center' }}>
                    <Typography variant="h6" color="text.secondary">
                      No videos in queue
                    </Typography>
                  </Paper>
                ) : (
                  filteredItems.map((item, index) => renderQueueItem(item, index))
                )}
                {provided.placeholder}
              </Box>
            )}
          </Droppable>
        </DragDropContext>
      )}
      {/* Action Menu */}
      <Menu
        anchorEl={anchorEl}
        open={Boolean(anchorEl)}
        onClose={() => setAnchorEl(null)}
      >
        <MenuItem
          onClick={() => {
            setEditDialogOpen(true);
            setAnchorEl(null);
          }}
        >
          <Edit sx={{ mr: 1 }} /> Edit
        </MenuItem>
        {selectedItem?.status === 'paused' ? (
          <MenuItem onClick={() => {
            handleResume(selectedItem.id);
            setAnchorEl(null);
          }}>
            <PlayArrow sx={{ mr: 1 }} /> Resume
          </MenuItem>
        ) : (
          <MenuItem onClick={() => {
            selectedItem && handlePause(selectedItem.id);
            setAnchorEl(null);
          }}>
            <Pause sx={{ mr: 1 }} /> Pause
          </MenuItem>
        )}
        {selectedItem?.status === 'failed' && (
          <MenuItem onClick={() => {
            handleRetry(selectedItem.id);
            setAnchorEl(null);
          }}>
            <Refresh sx={{ mr: 1 }} /> Retry
          </MenuItem>
        )}
        <MenuItem
          onClick={() => {
            selectedItem && handleDelete(selectedItem.id);
            setAnchorEl(null);
          }}
          sx={{ color: 'error.main' }}
        >
          <Delete sx={{ mr: 1 }} /> Delete
        </MenuItem>
      </Menu>
    </Box>
  );
};
