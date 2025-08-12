import React, { useState, useEffect } from 'react';
import {
  Box,
  Grid,
  Card,
  CardContent,
  CardHeader,
  Typography,
  Avatar,
  Button,
  IconButton,
  Chip,
  LinearProgress,
  Tab,
  Tabs,
  Paper,
  List,
  ListItem,
  ListItemAvatar,
  ListItemText,
  ListItemSecondaryAction,
  Divider,
  Menu,
  MenuItem,
  Dialog,
  DialogTitle,
  DialogContent,
  DialogActions,
  TextField,
  FormControl,
  InputLabel,
  Select,
  Switch,
  FormControlLabel,
  Alert,
  Tooltip,
  Badge,
  useTheme,
  useMediaQuery,
} from '@mui/material';
import {
  YouTube,
  Settings,
  MoreVert,
  Edit,
  Delete,
  Schedule,
  PlayCircle,
  Visibility,
  ThumbUp,
  Comment,
  TrendingUp,
  AttachMoney,
  Warning,
  CheckCircle,
  CloudUpload,
  Download,
  Share,
  ContentCopy,
  Refresh,
  BarChart,
  Timeline,
  VideoLibrary,
  Speed,
  HealthAndSafety,
  Assignment,
  FilterList,
  Sort,
  Search,
} from '@mui/icons-material';
import { format, formatDistanceToNow, subDays } from 'date-fns';
import { RealTimeMetrics } from '../Dashboard/RealTimeMetrics';
import { InlineHelp } from '../Common/InlineHelp';
import { HelpTooltip } from '../Common/HelpTooltip';

interface Channel {
  id: string;
  name: string;
  handle: string;
  thumbnail: string;
  subscribers: number;
  totalVideos: number;
  totalViews: number;
  monthlyRevenue: number;
  health: number;
  status: 'active' | 'paused' | 'warning' | 'error';
  lastVideoDate: Date;
  quotaUsage: number;
  quotaLimit: number;
  verified: boolean;
  monetized: boolean;
}

interface Video {
  id: string;
  title: string;
  thumbnail: string;
  status: 'published' | 'scheduled' | 'processing' | 'draft' | 'failed';
  publishedAt?: Date;
  scheduledAt?: Date;
  views: number;
  likes: number;
  comments: number;
  revenue: number;
  duration: string;
  ctr: number;
  avd: number;
}

interface ChannelMetric {
  label: string;
  value: string | number;
  change: number;
  icon: React.ReactNode;
  color: string;
}

interface TabPanelProps {
  children?: React.ReactNode;
  index: number;
  value: number;
}

const TabPanel: React.FC<TabPanelProps> = ({ children, value, index }) => {
  return (
    <div hidden={value !== index}>
      {value === index && <Box sx={{ py: 3 }}>{children}</Box>}
    </div>
  );
};

export const ChannelDashboard: React.FC<{ channelId: string }> = ({ channelId }) => {
  const theme = useTheme();
  const isMobile = useMediaQuery(theme.breakpoints.down('sm'));
  
  const [channel, setChannel] = useState<Channel | null>(null);
  const [videos, setVideos] = useState<Video[]>([]);
  const [tabValue, setTabValue] = useState(0);
  const [loading, setLoading] = useState(true);
  const [anchorEl, setAnchorEl] = useState<null | HTMLElement>(null);
  const [selectedVideo, setSelectedVideo] = useState<Video | null>(null);
  const [settingsOpen, setSettingsOpen] = useState(false);
  const [filterOpen, setFilterOpen] = useState(false);
  const [videoFilter, setVideoFilter] = useState('all');
  const [sortBy, setSortBy] = useState('date');

  // Mock data - replace with API calls
  useEffect(() => {
    // Simulate API call
    setTimeout(() => {
      setChannel({
        id: channelId,
        name: 'Tech Insights Daily',
        handle: '@techinsights',
        thumbnail: '/channel-thumb.jpg',
        subscribers: 125000,
        totalVideos: 342,
        totalViews: 8500000,
        monthlyRevenue: 3250.75,
        health: 85,
        status: 'active',
        lastVideoDate: new Date(),
        quotaUsage: 7500,
        quotaLimit: 10000,
        verified: true,
        monetized: true,
      });

      setVideos([
        {
          id: '1',
          title: '10 AI Tools That Will Change Your Life in 2024',
          thumbnail: '/video1.jpg',
          status: 'published',
          publishedAt: new Date(),
          views: 15234,
          likes: 892,
          comments: 156,
          revenue: 45.67,
          duration: '12:34',
          ctr: 4.5,
          avd: 65,
        },
        {
          id: '2',
          title: 'The Future of Quantum Computing Explained',
          thumbnail: '/video2.jpg',
          status: 'scheduled',
          scheduledAt: new Date(Date.now() + 86400000),
          views: 0,
          likes: 0,
          comments: 0,
          revenue: 0,
          duration: '15:22',
          ctr: 0,
          avd: 0,
        },
        {
          id: '3',
          title: 'Building a Smart Home on a Budget',
          thumbnail: '/video3.jpg',
          status: 'processing',
          views: 0,
          likes: 0,
          comments: 0,
          revenue: 0,
          duration: '10:15',
          ctr: 0,
          avd: 0,
        },
      ]);

      setLoading(false);
    }, 1000);
  }, [channelId]);

  const channelMetrics: ChannelMetric[] = [
    {
      label: 'Subscribers',
      value: channel?.subscribers.toLocaleString() || 0,
      change: 5.2,
      icon: <YouTube />,
      color: theme.palette.error.main,
    },
    {
      label: 'Total Views',
      value: channel?.totalViews.toLocaleString() || 0,
      change: 12.5,
      icon: <Visibility />,
      color: theme.palette.primary.main,
    },
    {
      label: 'Monthly Revenue',
      value: `$${channel?.monthlyRevenue.toFixed(2) || 0}`,
      change: 8.3,
      icon: <AttachMoney />,
      color: theme.palette.success.main,
    },
    {
      label: 'Channel Health',
      value: `${channel?.health || 0}%`,
      change: 2.1,
      icon: <Speed />,
      color: theme.palette.warning.main,
    },
  ];

  const getStatusColor = (status: string) => {
    switch (status) {
      case 'published':
      case 'active':
        return 'success';
      case 'scheduled':
        return 'info';
      case 'processing':
        return 'warning';
      case 'failed':
      case 'error':
        return 'error';
      default:
        return 'default';
    }
  };

  const handleVideoMenu = (event: React.MouseEvent<HTMLElement>, video: Video) => {
    setAnchorEl(event.currentTarget);
    setSelectedVideo(video);
  };

  const handleCloseMenu = () => {
    setAnchorEl(null);
    setSelectedVideo(null);
  };

  const renderChannelOverview = () => (
    <Grid container spacing={3}>
      {/* Channel Header */}
      <Grid item xs={12}>
        <Card>
          <CardContent>
            <Box sx={{ display: 'flex', alignItems: 'center', gap: 3, flexWrap: 'wrap' }}>
              <Avatar
                src={channel?.thumbnail}
                sx={{ width: 80, height: 80 }}
              >
                {channel?.name[0]}
              </Avatar>
              
              <Box sx={{ flex: 1 }}>
                <Box sx={{ display: 'flex', alignItems: 'center', gap: 1, mb: 1 }}>
                  <Typography variant="h5" fontWeight="bold">
                    {channel?.name}
                  </Typography>
                  {channel?.verified && (
                    <Tooltip title="Verified Channel">
                      <CheckCircle color="primary" fontSize="small" />
                    </Tooltip>
                  )}
                  {channel?.monetized && (
                    <Tooltip title="Monetization Enabled">
                      <AttachMoney color="success" fontSize="small" />
                    </Tooltip>
                  )}
                  <Chip
                    label={channel?.status}
                    size="small"
                    color={getStatusColor(channel?.status || '')}
                  />
                </Box>
                
                <Typography variant="body2" color="text.secondary" gutterBottom>
                  {channel?.handle} • {channel?.totalVideos} videos
                </Typography>
                
                <Box sx={{ display: 'flex', gap: 1, mt: 1 }}>
                  <Chip
                    icon={<Schedule />}
                    label={`Last video: ${channel?.lastVideoDate ? formatDistanceToNow(channel.lastVideoDate, { addSuffix: true }) : 'N/A'}`}
                    size="small"
                    variant="outlined"
                  />
                  <Chip
                    icon={<CloudUpload />}
                    label={`Quota: ${channel?.quotaUsage}/${channel?.quotaLimit}`}
                    size="small"
                    variant="outlined"
                    color={channel && channel.quotaUsage > channel.quotaLimit * 0.8 ? 'warning' : 'default'}
                  />
                </Box>
              </Box>
              
              <Box sx={{ display: 'flex', gap: 1 }}>
                <Button
                  variant="contained"
                  startIcon={<VideoLibrary />}
                  onClick={() => setTabValue(1)}
                >
                  Create Video
                </Button>
                <Button
                  variant="outlined"
                  startIcon={<Settings />}
                  onClick={() => setSettingsOpen(true)}
                >
                  Settings
                </Button>
                <IconButton onClick={(e) => setAnchorEl(e.currentTarget)}>
                  <MoreVert />
                </IconButton>
              </Box>
            </Box>
            
            {/* Quota Progress */}
            <Box sx={{ mt: 3 }}>
              <Box sx={{ display: 'flex', justifyContent: 'space-between', mb: 1 }}>
                <Typography variant="body2">API Quota Usage</Typography>
                <Typography variant="body2" fontWeight="bold">
                  {channel ? ((channel.quotaUsage / channel.quotaLimit) * 100).toFixed(1) : 0}%
                </Typography>
              </Box>
              <LinearProgress
                variant="determinate"
                value={channel ? (channel.quotaUsage / channel.quotaLimit) * 100 : 0}
                color={channel && channel.quotaUsage > channel.quotaLimit * 0.8 ? 'warning' : 'primary'}
              />
            </Box>
          </CardContent>
        </Card>
      </Grid>

      {/* Key Metrics */}
      {channelMetrics.map((metric, index) => (
        <Grid item xs={12} sm={6} md={3} key={index}>
          <Card>
            <CardContent>
              <Box sx={{ display: 'flex', alignItems: 'center', mb: 2 }}>
                <Avatar
                  sx={{
                    bgcolor: `${metric.color}20`,
                    color: metric.color,
                    width: 40,
                    height: 40,
                  }}
                >
                  {metric.icon}
                </Avatar>
                <Box sx={{ ml: 'auto' }}>
                  <Chip
                    label={`+${metric.change}%`}
                    size="small"
                    color="success"
                    sx={{ fontSize: 11 }}
                  />
                </Box>
              </Box>
              <Typography variant="h5" fontWeight="bold">
                {metric.value}
              </Typography>
              <Typography variant="body2" color="text.secondary">
                {metric.label}
              </Typography>
            </CardContent>
          </Card>
        </Grid>
      ))}

      {/* Real-time Metrics */}
      <Grid item xs={12}>
        <Typography variant="h6" fontWeight="bold" gutterBottom>
          Real-time Performance
        </Typography>
        <RealTimeMetrics
          channelId={channelId}
          compactMode={isMobile}
          showSparklines={!isMobile}
        />
      </Grid>

      {/* Help Section */}
      <Grid item xs={12}>
        <InlineHelp
          context="channel-management"
          variant="compact"
        />
      </Grid>
    </Grid>
  );

  const renderVideoHistory = () => (
    <Box>
      {/* Filters and Actions */}
      <Box sx={{ display: 'flex', gap: 2, mb: 3, flexWrap: 'wrap' }}>
        <TextField
          placeholder="Search videos..."
          size="small"
          InputProps={{
            startAdornment: <Search sx={{ mr: 1, color: 'text.secondary' }} />,
          }}
          sx={{ flex: 1, minWidth: 200 }}
        />
        
        <FormControl size="small" sx={{ minWidth: 120 }}>
          <InputLabel>Status</InputLabel>
          <Select
            value={videoFilter}
            onChange={(e) => setVideoFilter(e.target.value)}
            label="Status"
          >
            <MenuItem value="all">All</MenuItem>
            <MenuItem value="published">Published</MenuItem>
            <MenuItem value="scheduled">Scheduled</MenuItem>
            <MenuItem value="processing">Processing</MenuItem>
            <MenuItem value="draft">Draft</MenuItem>
          </Select>
        </FormControl>
        
        <FormControl size="small" sx={{ minWidth: 120 }}>
          <InputLabel>Sort By</InputLabel>
          <Select
            value={sortBy}
            onChange={(e) => setSortBy(e.target.value)}
            label="Sort By"
          >
            <MenuItem value="date">Date</MenuItem>
            <MenuItem value="views">Views</MenuItem>
            <MenuItem value="revenue">Revenue</MenuItem>
            <MenuItem value="engagement">Engagement</MenuItem>
          </Select>
        </FormControl>
        
        <Button
          variant="outlined"
          startIcon={<Refresh />}
        >
          Refresh
        </Button>
      </Box>

      {/* Video List */}
      <List>
        {videos.map((video, index) => (
          <React.Fragment key={video.id}>
            <ListItem alignItems="flex-start">
              <ListItemAvatar>
                <Avatar
                  variant="rounded"
                  src={video.thumbnail}
                  sx={{ width: 120, height: 67.5, mr: 2 }}
                >
                  <PlayCircle />
                </Avatar>
              </ListItemAvatar>
              
              <ListItemText
                primary={
                  <Box sx={{ display: 'flex', alignItems: 'center', gap: 1 }}>
                    <Typography variant="subtitle1" fontWeight="medium">
                      {video.title}
                    </Typography>
                    <Chip
                      label={video.status}
                      size="small"
                      color={getStatusColor(video.status)}
                    />
                  </Box>
                }
                secondary={
                  <Box>
                    <Typography variant="body2" color="text.secondary">
                      {video.status === 'published' && video.publishedAt
                        ? `Published ${formatDistanceToNow(video.publishedAt, { addSuffix: true })}`
                        : video.status === 'scheduled' && video.scheduledAt
                        ? `Scheduled for ${format(video.scheduledAt, 'PPp')}`
                        : `Status: ${video.status}`}
                    </Typography>
                    
                    {video.status === 'published' && (
                      <Box sx={{ display: 'flex', gap: 2, mt: 1 }}>
                        <Chip
                          icon={<Visibility />}
                          label={video.views.toLocaleString()}
                          size="small"
                          variant="outlined"
                        />
                        <Chip
                          icon={<ThumbUp />}
                          label={video.likes.toLocaleString()}
                          size="small"
                          variant="outlined"
                        />
                        <Chip
                          icon={<Comment />}
                          label={video.comments.toLocaleString()}
                          size="small"
                          variant="outlined"
                        />
                        <Chip
                          icon={<AttachMoney />}
                          label={`$${video.revenue.toFixed(2)}`}
                          size="small"
                          variant="outlined"
                          color="success"
                        />
                      </Box>
                    )}
                    
                    {video.status === 'published' && (
                      <Box sx={{ display: 'flex', gap: 2, mt: 1 }}>
                        <Typography variant="caption" color="text.secondary">
                          CTR: {video.ctr}%
                        </Typography>
                        <Typography variant="caption" color="text.secondary">
                          AVD: {video.avd}%
                        </Typography>
                        <Typography variant="caption" color="text.secondary">
                          Duration: {video.duration}
                        </Typography>
                      </Box>
                    )}
                  </Box>
                }
              />
              
              <ListItemSecondaryAction>
                <IconButton
                  edge="end"
                  onClick={(e) => handleVideoMenu(e, video)}
                >
                  <MoreVert />
                </IconButton>
              </ListItemSecondaryAction>
            </ListItem>
            {index < videos.length - 1 && <Divider component="li" />}
          </React.Fragment>
        ))}
      </List>

      {/* Video Actions Menu */}
      <Menu
        anchorEl={anchorEl}
        open={Boolean(anchorEl) && Boolean(selectedVideo)}
        onClose={handleCloseMenu}
      >
        <MenuItem onClick={handleCloseMenu}>
          <ListItemIcon>
            <Edit fontSize="small" />
          </ListItemIcon>
          <ListItemText>Edit</ListItemText>
        </MenuItem>
        <MenuItem onClick={handleCloseMenu}>
          <ListItemIcon>
            <ContentCopy fontSize="small" />
          </ListItemIcon>
          <ListItemText>Duplicate</ListItemText>
        </MenuItem>
        <MenuItem onClick={handleCloseMenu}>
          <ListItemIcon>
            <Share fontSize="small" />
          </ListItemIcon>
          <ListItemText>Share</ListItemText>
        </MenuItem>
        <MenuItem onClick={handleCloseMenu}>
          <ListItemIcon>
            <Download fontSize="small" />
          </ListItemIcon>
          <ListItemText>Download</ListItemText>
        </MenuItem>
        <Divider />
        <MenuItem onClick={handleCloseMenu} sx={{ color: 'error.main' }}>
          <ListItemIcon>
            <Delete fontSize="small" color="error" />
          </ListItemIcon>
          <ListItemText>Delete</ListItemText>
        </MenuItem>
      </Menu>
    </Box>
  );

  const renderChannelSettings = () => (
    <Grid container spacing={3}>
      <Grid item xs={12} md={6}>
        <Card>
          <CardHeader title="Channel Configuration" />
          <CardContent>
            <Box sx={{ display: 'flex', flexDirection: 'column', gap: 2 }}>
              <TextField
                label="Channel Name"
                value={channel?.name || ''}
                fullWidth
              />
              <TextField
                label="Channel Handle"
                value={channel?.handle || ''}
                fullWidth
              />
              <TextField
                label="Description"
                multiline
                rows={4}
                fullWidth
              />
              <FormControlLabel
                control={<Switch checked={channel?.monetized || false} />}
                label="Monetization Enabled"
              />
              <FormControlLabel
                control={<Switch checked={true} />}
                label="Auto-publish Videos"
              />
              <Button variant="contained">
                Save Changes
              </Button>
            </Box>
          </CardContent>
        </Card>
      </Grid>

      <Grid item xs={12} md={6}>
        <Card>
          <CardHeader title="Upload Defaults" />
          <CardContent>
            <Box sx={{ display: 'flex', flexDirection: 'column', gap: 2 }}>
              <FormControl fullWidth>
                <InputLabel>Default Category</InputLabel>
                <Select value="technology" label="Default Category">
                  <MenuItem value="technology">Science & Technology</MenuItem>
                  <MenuItem value="education">Education</MenuItem>
                  <MenuItem value="entertainment">Entertainment</MenuItem>
                </Select>
              </FormControl>
              <TextField
                label="Default Tags"
                placeholder="tech, ai, innovation"
                fullWidth
              />
              <FormControl fullWidth>
                <InputLabel>Default Visibility</InputLabel>
                <Select value="public" label="Default Visibility">
                  <MenuItem value="public">Public</MenuItem>
                  <MenuItem value="unlisted">Unlisted</MenuItem>
                  <MenuItem value="private">Private</MenuItem>
                </Select>
              </FormControl>
              <FormControlLabel
                control={<Switch checked={true} />}
                label="Enable Comments"
              />
              <FormControlLabel
                control={<Switch checked={true} />}
                label="Enable Likes"
              />
            </Box>
          </CardContent>
        </Card>
      </Grid>

      <Grid item xs={12}>
        <Alert severity="info">
          Changes to channel settings may take up to 24 hours to fully propagate across YouTube's systems.
        </Alert>
      </Grid>
    </Grid>
  );

  if (loading) {
    return (
      <Box sx={{ display: 'flex', justifyContent: 'center', p: 4 }}>
        <CircularProgress />
      </Box>
    );
  }

  return (
    <Box>
      <Tabs
        value={tabValue}
        onChange={(e, newValue) => setTabValue(newValue)}
        variant={isMobile ? 'scrollable' : 'standard'}
        scrollButtons={isMobile ? 'auto' : false}
      >
        <Tab label="Overview" icon={<Dashboard />} iconPosition="start" />
        <Tab label="Videos" icon={<VideoLibrary />} iconPosition="start" />
        <Tab label="Analytics" icon={<BarChart />} iconPosition="start" />
        <Tab label="Settings" icon={<Settings />} iconPosition="start" />
      </Tabs>

      <TabPanel value={tabValue} index={0}>
        {renderChannelOverview()}
      </TabPanel>

      <TabPanel value={tabValue} index={1}>
        {renderVideoHistory()}
      </TabPanel>

      <TabPanel value={tabValue} index={2}>
        <Typography>Analytics content coming soon...</Typography>
      </TabPanel>

      <TabPanel value={tabValue} index={3}>
        {renderChannelSettings()}
      </TabPanel>
    </Box>
  );
};