/**
 * Video Queue Interface Component
 * MVP Screen Design - Video queue management
 */
import React, { useState, useEffect } from 'react';
import { 
  Box,
  Paper,
  Typography,
  Button,
  Card,
  CardContent,
  IconButton,
  Chip,
  Avatar,
  ListItemAvatar,
  LinearProgress,
  Menu,
  MenuItem,
  Dialog,
  DialogTitle,
  DialogContent,
  DialogActions,
  Tabs,
  Tab,
  Alert,
  Grid,
  ToggleButton,
  ToggleButtonGroup,
  Checkbox
 } from '@mui/material';
import { 
  Queue,
  PlayArrow,
  Pause,
  Delete,
  Edit,
  Schedule,
  CheckCircle,
  AttachMoney,
  Visibility,
  Refresh,
  FilterList,
  Sort,
  DragIndicator,
  AutorenewOutlined,
  ViewList,
  ViewModule
 } from '@mui/icons-material';
import {  DragDropContext, Droppable, Draggable  } from 'react-beautiful-dnd';

interface Video {
  id: string,
  title: string,

  channel: string,
  channelId: string,

  status: 'queued' | 'processing' | 'completed' | 'failed' | 'scheduled',
  progress: number,

  thumbnail: string,
  duration: string;
  scheduledDate?: Date;
  priority: 'low' | 'normal' | 'high' | 'urgent',
  cost: number,

  estimatedViews: number,
  tags: string[],

  createdAt: Date;
  processingStage?: string;
  error?: string;
}

const mockVideos: Video[] = [
  { id: '1',
    title: 'Top 10 JavaScript Frameworks in 2024',
    channel: 'Tech Reviews Pro',
    channelId: '1',
    status: 'processing',
    progress: 65,
    thumbnail: '/api/placeholder/160/90',
    duration: '12:34',
    priority: 'high',
    cost: 2.85,
    estimatedViews: 25000,
    tags: ['JavaScript', 'Programming', 'Tutorial'],
    createdAt: new Date(),
    processingStage: 'Generating voice narration' },
  { id: '2',
    title: 'Ultimate Gaming PC Build Guide',
    channel: 'Gaming Highlights',
    channelId: '2',
    status: 'queued',
    progress: 0,
    thumbnail: '/api/placeholder/160/90',
    duration: '15:20',
    priority: 'normal',
    cost: 3.20,
    estimatedViews: 18000,
    tags: ['Gaming', 'PC Build', 'Hardware'],
    createdAt: new Date() },
  { id: '3',
    title: '5 Easy Pasta Recipes for Beginners',
    channel: 'Cooking Adventures',
    channelId: '3',
    status: 'scheduled',
    progress: 100,
    thumbnail: '/api/placeholder/160/90',
    duration: '8:45',
    scheduledDate: new Date(Date.now() + 3600000),
    priority: 'normal',
    cost: 1.95,
    estimatedViews: 12000,
    tags: ['Cooking', 'Recipe', 'Food'],
    createdAt: new Date() },
  { id: '4',
    title: 'React vs Vue.js - Performance Comparison',
    channel: 'Tech Reviews Pro',
    channelId: '1',
    status: 'completed',
    progress: 100,
    thumbnail: '/api/placeholder/160/90',
    duration: '10:15',
    priority: 'normal',
    cost: 2.45,
    estimatedViews: 22000,
    tags: ['React', 'Vue', 'Comparison'],
    createdAt: new Date() },
  { id: '5',
    title: 'Best RPG Games of 2024',
    channel: 'Gaming Highlights',
    channelId: '2',
    status: 'failed',
    progress: 35,
    thumbnail: '/api/placeholder/160/90',
    duration: '18:00',
    priority: 'low',
    cost: 0.85,
    estimatedViews: 30000,
    tags: ['Gaming', 'RPG', 'Review'],
    createdAt: new Date(),
    _: 'Voice synthesis failed: API quota exceeded' }];

export const VideoQueue: React.FC = () => {
  const [videos, setVideos] = useState<Video[]>(mockVideos);
  const [tabValue, setTabValue] = useState(0);
  const [viewMode, setViewMode] = useState<'list' | 'grid'>('list');
  const [selectedVideos, setSelectedVideos] = useState<string[]>([]);
  const [filterMenu, setFilterMenu] = useState<null | HTMLElement>(null);
  const [filterStatus, setFilterStatus] = useState('all');
  const [detailDialog, setDetailDialog] = useState<Video | null>(null);

  const handleTabChange = (_: React.SyntheticEvent, newValue: number) => {
    setTabValue(newValue)};

  const handleDragEnd = (result: React.ChangeEvent<HTMLInputElement>) => {
    if (!result.destination) return;
    
    const items = Array.from(videos);
    const [reorderedItem] = items.splice(result.source.index, 1);
    items.splice(result.destination.index, 0, reorderedItem);
    
    setVideos(items)};

  const getStatusColor = (status: string) => {
    switch (status) {
      case 'completed': return 'success';
      case 'processing': return 'info';
      case 'queued': return 'default';
      case 'scheduled': return 'warning';
      case 'failed': return 'error';
      default: return 'default'}
  };

  const getStatusIcon = (status: string) => {
    switch (status) {
      case 'completed': return <CheckCircle />;
      case 'processing': return <AutorenewOutlined />;
      case 'queued': return <Queue />;
      case 'scheduled': return <Schedule />;
      case 'failed': return <ErrorIcon />;
      default: return null}
  };

  const getPriorityColor = (priority: string) => {
    switch (priority) {
      case 'urgent': return 'error';
      case 'high': return 'warning';
      case 'normal': return 'info';
      case 'low': return 'default';
      default: return 'default'}
  };

  const getFilteredVideos = () => {
    let filtered = [...videos];
    
    if (filterStatus !== 'all') {
      filtered = filtered.filter(v => v.status === filterStatus)}
    
    if (tabValue === 1) filtered = filtered.filter(v => v.status === 'queued');
    if (tabValue === 2) filtered = filtered.filter(v => v.status === 'processing');
    if (tabValue === 3) filtered = filtered.filter(v => v.status === 'completed');
    if (tabValue === 4) filtered = filtered.filter(v => v.status === 'failed');
    
    // Sort
    filtered.sort(_(a, _b) => {
      if (sortBy === 'priority') {
        const priorityOrder = { urgent: 0, high: 1, normal: 2, low: 3 };
        return priorityOrder[a.priority] - priorityOrder[b.priority];
      }
      if (sortBy === 'date') {
        return b.createdAt.getTime() - a.createdAt.getTime()}
      if (sortBy === 'cost') {
        return b.cost - a.cost;
      }
      return 0});
    
    return filtered;
  };

  const handleBulkAction = (action: string) => {
    switch (action) {
      case 'pause':
        // Pause selected videos
        break;
      case 'resume':
        // Resume selected videos
        break;
      case 'delete':
        // Delete selected videos
        setVideos(videos.filter(v => !selectedVideos.includes(v.id)));
        setSelectedVideos([]);
        break;
    }
  };

  const VideoListItem = ({ video, index }: { video: Video; index: number }) => (
    <Draggable draggableId={video.id} index={index}>
      {(provided) => (
        <ListItem
          ref={provided.innerRef}
          {...provided.draggableProps}
          sx={{
            mb: 2,
            bgcolor: 'background.paper',
            borderRadius: 2,
            border: '1px solid',
            borderColor: 'divider',
            '&:hover': { bgcolor: 'action.hover' }
          }}
        >
          <IconButton {...provided.dragHandleProps} size="small" sx={{ mr: 1 }}>
            <DragIndicator />
          </IconButton>
          
          <Checkbox
            checked={selectedVideos.includes(video.id)}
            onChange={(_) => {
              if (e.target.checked) {
                setSelectedVideos([...selectedVideos, video.id])} else {
                setSelectedVideos(selectedVideos.filter(id => id !== video.id))}
            }}
            sx={{ mr: 1 }}
          />
          
          <ListItemAvatar>
            <Box sx={{ position: 'relative' }}>
              <Avatar
                variant="rounded"
                src={video.thumbnail}
                sx={{ width: 120, height: 68 }}
              />
              <Typography
                variant="caption"
                sx={ {
                  position: 'absolute',
                  bottom: 4,
                  right: 4,
                  bgcolor: 'rgba(0,0,0,0.7)',
                  color: 'white',
                  px: 0.5,
                  borderRadius: 0.5 }}
              >
                {video.duration}
              </Typography>
            </Box>
          </ListItemAvatar>
          
          <ListItemText
            primary={
              <Box sx={{ display: 'flex', alignItems: 'center', gap: 1 }}>
                <Typography variant="subtitle1">{video.title}</Typography>
                <Chip
                  size="small"
                  label={video.priority}
                  color={getPriorityColor(video.priority) as any}
                />
                <Chip
                  size="small"
                  icon={getStatusIcon(video.status)}
                  label={video.status}
                  color={getStatusColor(video.status) as any}
                />
              </Box>
            }
            secondary={
              <Box>
                <Typography variant="body2" color="text.secondary">
                  {video.channel} • Created {video.createdAt.toLocaleDateString()}
                </Typography>
                {video.status === 'processing' && (
                  <Box sx={{ mt: 1 }}>
                    <Typography variant="caption" color="primary">
                      {video.processingStage}
                    </Typography>
                    <LinearProgress
                      variant="determinate"
                      value={video.progress}
                      sx={{ mt: 0.5 }}
                    />
                  </Box>
                )}
                {video.status === 'failed' && (
                  <Alert severity="error" sx={{ mt: 1, py: 0 }}>
                    {video.error}
                  </Alert>
                )}
                <Box sx={{ display: 'flex', gap: 2, mt: 1 }}>
                  <Typography variant="caption" sx={{ display: 'flex', alignItems: 'center', gap: 0.5 }}>
                    <AttachMoney fontSize="small" /> ${video.cost}
                  </Typography>
                  <Typography variant="caption" sx={{ display: 'flex', alignItems: 'center', gap: 0.5 }}>
                    <Visibility fontSize="small" /> {video.estimatedViews.toLocaleString()} est. views
                  </Typography>
                  {video.scheduledDate && (
                    <Typography variant="caption" sx={{ display: 'flex', alignItems: 'center', gap: 0.5 }}>
                      <Schedule fontSize="small" /> {video.scheduledDate.toLocaleString()}
                    </Typography>
                  )}
                </Box>
                <Box sx={{ mt: 1 }}>
                  {video.tags.map((tag) => (
                    <Chip key={tag} label={tag} size="small" sx={{ mr: 0.5 }} />
                  ))}
                </Box>
              </Box>
            }
          />
          
          <ListItemSecondaryAction>
            <Stack direction="row" spacing={1}>
              {video.status === 'queued' && (
                <IconButton size="small" color="primary">
                  <PlayArrow />
                </IconButton>
              )}
              {video.status === 'processing' && (
                <IconButton size="small" color="warning">
                  <Pause />
                </IconButton>
              )}
              {video.status === 'failed' && (
                <IconButton size="small" color="info">
                  <Refresh />
                </IconButton>
              )}
              <IconButton size="small" onClick={() => setDetailDialog(video}>
                <Edit />
              </IconButton>
              <IconButton size="small" color="error">
                <Delete />
              </IconButton>
            </Stack>
          </ListItemSecondaryAction>
        </ListItem>
      )}
    </Draggable>
  );

  return (
    <>
      <Box sx={{ flexGrow: 1, p: 3 }}>
      {/* Header */}
      <Box sx={{ mb: 3, display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
        <Box>
          <Typography variant="h4" gutterBottom>
            Video Queue
          </Typography>
      <Typography variant="body2" color="text.secondary">
            Manage and monitor your video generation pipeline
          </Typography>
        </Box>
        <Stack direction="row" spacing={2}>
          <Button variant="outlined" startIcon={<Refresh />}>
            Refresh
          </Button>
          <Button variant="contained" startIcon={<PlayArrow />}>
            Process All
          </Button>
        </Stack>
      </Box>

      {/* Stats Cards */}
      <Grid container spacing={2} sx={{ mb: 3 }}>
        <Grid item xs={6} sm={3}>
          <Card>
            <CardContent>
              <Typography color="text.secondary" gutterBottom>
                Queued
              </Typography>
              <Typography variant="h5">
                {videos.filter(v => v.status === 'queued').length}
              </Typography>
            </CardContent>
          </Card>
        </Grid>
        <Grid item xs={6} sm={3}>
          <Card>
            <CardContent>
              <Typography color="text.secondary" gutterBottom>
                Processing
              </Typography>
              <Typography variant="h5">
                {videos.filter(v => v.status === 'processing').length}
              </Typography>
            </CardContent>
          </Card>
        </Grid>
        <Grid item xs={6} sm={3}>
          <Card>
            <CardContent>
              <Typography color="text.secondary" gutterBottom>
                Completed Today
              </Typography>
              <Typography variant="h5">
                {videos.filter(v => v.status === 'completed').length}
              </Typography>
            </CardContent>
          </Card>
        </Grid>
        <Grid item xs={6} sm={3}>
          <Card>
            <CardContent>
              <Typography color="text.secondary" gutterBottom>
                Failed
              </Typography>
              <Typography variant="h5" color="error">
                {videos.filter(v => v.status === 'failed').length}
              </Typography>
            </CardContent>
          </Card>
        </Grid>
      </Grid>

      {/* Toolbar */}
      <Paper sx={{ p: 2, mb: 2 }}>
        <Box sx={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
          <Tabs value={tabValue} onChange={handleTabChange}>
            <Tab label="All" />
            <Tab 
              label={
                <Badge badgeContent={videos.filter(v => v.status === 'queued').length} color="default">
                  Queued
                </Badge>
              }
            />
            <Tab
              label={
                <Badge badgeContent={videos.filter(v => v.status === 'processing').length} color="info">
                  Processing
                </Badge>
              }
            />
            <Tab label="Completed" />
            <Tab
              label={
                <Badge badgeContent={videos.filter(v => v.status === 'failed').length} color="error">
                  Failed
                </Badge>
              }
            />
          </Tabs>
          
          <Stack direction="row" spacing={1}>
            {selectedVideos.length > 0 && (
              <>
                <Button size="small" onClick={() => handleBulkAction('pause'}>
                  Pause Selected
                </Button>
                <Button size="small" color="error" onClick={() => handleBulkAction('delete'}>
                  Delete Selected
                </Button>
                <Divider orientation="vertical" flexItem />
              </>
            )}
            <ToggleButtonGroup
              value={viewMode}
              exclusive
              onChange={(_, newMode) => newMode && setViewMode(newMode}
              size="small"
            >
              <ToggleButton value="list">
                <ViewList />
              </ToggleButton>
              <ToggleButton value="grid">
                <ViewModule />
              </ToggleButton>
            </ToggleButtonGroup>
            <IconButton size="small" onClick={(_) => setFilterMenu(_.currentTarget}>
              <FilterList />
            </IconButton>
            <IconButton size="small">
              <Sort />
            </IconButton>
          </Stack>
        </Box>
      </Paper>

      {/* Video List */}
      <DragDropContext onDragEnd={handleDragEnd}>
        <Droppable droppableId="videos">
          {(provided) => (
            <List {...provided.droppableProps} ref={provided.innerRef}>
              {getFilteredVideos().map((video, index) => (
                <VideoListItem key={video.id} video={video} index={index} />
              ))}
              {provided.placeholder}
            </List>
          )}
        </Droppable>
      </DragDropContext>

      {/* Filter Menu */}
      <Menu
        anchorEl={filterMenu}
        open={Boolean(filterMenu)}
        onClose={() => setFilterMenu(null}
      >
        <MenuItem onClick={(</>
  ) => { setFilterStatus('all'</>
  ); setFilterMenu(null</>
  )}}>
          All Status
        </MenuItem>
        <MenuItem onClick={() => { setFilterStatus('queued'); setFilterMenu(null)}}>
          Queued Only
        </MenuItem>
        <MenuItem onClick={() => { setFilterStatus('processing'); setFilterMenu(null)}}>
          Processing Only
        </MenuItem>
        <MenuItem onClick={() => { setFilterStatus('completed'); setFilterMenu(null)}}>
          Completed Only
        </MenuItem>
        <MenuItem onClick={() => { setFilterStatus('failed'); setFilterMenu(null)}}>
          Failed Only
        </MenuItem>
      </Menu>

      {/* Detail Dialog */}
      <Dialog open={Boolean(detailDialog)} onClose={() => setDetailDialog(null} maxWidth="md" fullWidth>
        {detailDialog && (
          <>
            <DialogTitle>{detailDialog.title}</DialogTitle>
            <DialogContent>
              {/* Video details form */}
            </DialogContent>
            <DialogActions>
              <Button onClick={() => setDetailDialog(null}>Close</Button>
              <Button variant="contained">Save Changes</Button>
            </DialogActions>
          </>
        )}
      </Dialog>
    </Box>
  )};