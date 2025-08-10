import React, { useEffect, useState } from 'react';
import {
  Box,
  Grid,
  Paper,
  Typography,
  Card,
  CardContent,
  Button,
  IconButton,
  LinearProgress,
  Chip,
  Avatar,
  List,
  ListItem,
  ListItemAvatar,
  ListItemText,
  Skeleton,
  Alert,
  useTheme,
  useMediaQuery,
} from '@mui/material';
import {
  TrendingUp,
  VideoLibrary,
  MonetizationOn,
  Visibility,
  Schedule,
  PlayCircleOutline,
  Add,
  Refresh,
  ArrowUpward,
  ArrowDownward,
  YouTube,
  Analytics as AnalyticsIcon,
  AutoAwesome,
  Warning,
  CheckCircle,
} from '@mui/icons-material';
import { Line, Bar, Doughnut } from 'react-chartjs-2';
import {
  Chart as ChartJS,
  CategoryScale,
  LinearScale,
  PointElement,
  LineElement,
  BarElement,
  ArcElement,
  Title,
  Tooltip,
  Legend,
  Filler,
} from 'chart.js';
import { useNavigate } from 'react-router-dom';
import { formatNumber, formatCurrency, formatDuration } from '../../utils/formatters';
import { dashboardApi } from '../../services/api';
import { useWebSocket } from '../../hooks/useWebSocket';

// Register ChartJS components
ChartJS.register(
  CategoryScale,
  LinearScale,
  PointElement,
  LineElement,
  BarElement,
  ArcElement,
  Title,
  Tooltip,
  Legend,
  Filler
);

interface DashboardStats {
  totalChannels: number;
  totalVideos: number;
  totalViews: number;
  totalRevenue: number;
  totalCost: number;
  profit: number;
  avgEngagementRate: number;
  videosInQueue: number;
  videosPublishedToday: number;
  monthlyGrowthRate: number;
  bestPerformingVideo: {
    id: string;
    title: string;
    views: number;
    revenue: number;
  } | null;
}

interface RecentActivity {
  id: string;
  type: 'video_generated' | 'video_published' | 'channel_connected' | 'milestone_reached';
  title: string;
  description: string;
  timestamp: string;
  icon: React.ReactNode;
  color: string;
}

interface VideoInQueue {
  id: string;
  title: string;
  channel: string;
  status: 'generating' | 'scheduled' | 'ready';
  progress?: number;
  scheduledTime?: string;
  thumbnail?: string;
}

export const MainDashboard: React.FC = () => {
  const theme = useTheme();
  const navigate = useNavigate();
  const isMobile = useMediaQuery(theme.breakpoints.down('sm'));
  const isTablet = useMediaQuery(theme.breakpoints.down('md'));
  
  const [stats, setStats] = useState<DashboardStats | null>(null);
  const [recentActivity, setRecentActivity] = useState<RecentActivity[]>([]);
  const [videosInQueue, setVideosInQueue] = useState<VideoInQueue[]>([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [refreshing, setRefreshing] = useState(false);

  // WebSocket for real-time updates
  const { data: wsData, isConnected } = useWebSocket('/dashboard');

  useEffect(() => {
    fetchDashboardData();
  }, []);

  useEffect(() => {
    if (wsData) {
      // Handle real-time updates
      handleRealtimeUpdate(wsData);
    }
  }, [wsData]);

  const fetchDashboardData = async () => {
    try {
      setLoading(true);
      const [statsRes, activityRes, queueRes] = await Promise.all([
        dashboardApi.getStats(),
        dashboardApi.getRecentActivity(),
        dashboardApi.getVideoQueue(),
      ]);
      
      setStats(statsRes.data);
      setRecentActivity(formatActivity(activityRes.data));
      setVideosInQueue(queueRes.data);
      setError(null);
    } catch (error: any) {
      setError('Failed to load dashboard data. Please refresh to try again.');
      console.error('Dashboard error:', error);
    } finally {
      setLoading(false);
    }
  };

  const handleRefresh = async () => {
    setRefreshing(true);
    await fetchDashboardData();
    setRefreshing(false);
  };

  const handleRealtimeUpdate = (data: any) => {
    if (data.type === 'stats_update') {
      setStats(data.stats);
    } else if (data.type === 'new_activity') {
      setRecentActivity(prev => [formatActivityItem(data.activity), ...prev].slice(0, 10));
    } else if (data.type === 'queue_update') {
      setVideosInQueue(data.queue);
    }
  };

  const formatActivity = (activities: any[]): RecentActivity[] => {
    return activities.map(formatActivityItem);
  };

  const formatActivityItem = (activity: any): RecentActivity => {
    const typeConfig = {
      video_generated: {
        icon: <AutoAwesome />,
        color: '#4caf50',
      },
      video_published: {
        icon: <PlayCircleOutline />,
        color: '#2196f3',
      },
      channel_connected: {
        icon: <YouTube />,
        color: '#f44336',
      },
      milestone_reached: {
        icon: <TrendingUp />,
        color: '#ff9800',
      },
    };

    const config = typeConfig[activity.type as keyof typeof typeConfig];
    
    return {
      ...activity,
      icon: config.icon,
      color: config.color,
    };
  };

  // Chart data
  const revenueChartData = {
    labels: ['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun'],
    datasets: [
      {
        label: 'Revenue',
        data: [120, 150, 180, 200, 170, 220, 250],
        borderColor: '#4caf50',
        backgroundColor: 'rgba(76, 175, 80, 0.1)',
        fill: true,
        tension: 0.4,
      },
      {
        label: 'Cost',
        data: [30, 35, 40, 38, 42, 45, 48],
        borderColor: '#f44336',
        backgroundColor: 'rgba(244, 67, 54, 0.1)',
        fill: true,
        tension: 0.4,
      },
    ],
  };

  const viewsChartData = {
    labels: ['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun'],
    datasets: [
      {
        label: 'Views',
        data: [1200, 1900, 3000, 5000, 4000, 3000, 4500],
        backgroundColor: 'rgba(103, 126, 234, 0.8)',
      },
    ],
  };

  const categoryChartData = {
    labels: ['Gaming', 'Education', 'Tech', 'Entertainment', 'Music'],
    datasets: [
      {
        data: [30, 25, 20, 15, 10],
        backgroundColor: [
          '#e91e63',
          '#2196f3',
          '#9c27b0',
          '#ff9800',
          '#4caf50',
        ],
      },
    ],
  };

  const chartOptions = {
    responsive: true,
    maintainAspectRatio: false,
    plugins: {
      legend: {
        display: false,
      },
    },
    scales: {
      y: {
        beginAtZero: true,
        grid: {
          display: false,
        },
      },
      x: {
        grid: {
          display: false,
        },
      },
    },
  };

  if (loading) {
    return (
      <Box>
        <Grid container spacing={3}>
          {[1, 2, 3, 4].map((i) => (
            <Grid item xs={12} sm={6} md={3} key={i}>
              <Skeleton variant="rectangular" height={120} />
            </Grid>
          ))}
          <Grid item xs={12} md={8}>
            <Skeleton variant="rectangular" height={400} />
          </Grid>
          <Grid item xs={12} md={4}>
            <Skeleton variant="rectangular" height={400} />
          </Grid>
        </Grid>
      </Box>
    );
  }

  if (error) {
    return (
      <Alert severity="error" action={
        <Button color="inherit" size="small" onClick={handleRefresh}>
          Retry
        </Button>
      }>
        {error}
      </Alert>
    );
  }

  return (
    <Box>
      {/* Header */}
      <Box sx={{ display: 'flex', justifyContent: 'space-between', mb: 3 }}>
        <Box>
          <Typography variant="h4" fontWeight="bold" gutterBottom>
            Dashboard
          </Typography>
          <Typography variant="body2" color="text.secondary">
            Welcome back! Here's what's happening with your channels.
          </Typography>
        </Box>
        <Box sx={{ display: 'flex', gap: 2, alignItems: 'center' }}>
          {isConnected && (
            <Chip
              icon={<CheckCircle />}
              label="Live"
              color="success"
              size="small"
            />
          )}
          <IconButton onClick={handleRefresh} disabled={refreshing}>
            <Refresh className={refreshing ? 'spinning' : ''} />
          </IconButton>
          <Button
            variant="contained"
            startIcon={<Add />}
            onClick={() => navigate('/videos/generate')}
            sx={{
              background: 'linear-gradient(135deg, #667eea 0%, #764ba2 100%)',
              '&:hover': {
                background: 'linear-gradient(135deg, #5a6fd8 0%, #6a4290 100%)',
              },
            }}
          >
            Generate Video
          </Button>
        </Box>
      </Box>

      {/* Stats Cards */}
      <Grid container spacing={3} sx={{ mb: 3 }}>
        <Grid item xs={12} sm={6} md={3}>
          <Card>
            <CardContent>
              <Box sx={{ display: 'flex', justifyContent: 'space-between', mb: 2 }}>
                <Box>
                  <Typography color="text.secondary" variant="body2">
                    Total Revenue
                  </Typography>
                  <Typography variant="h5" fontWeight="bold">
                    {formatCurrency(stats?.totalRevenue || 0)}
                  </Typography>
                </Box>
                <Avatar sx={{ bgcolor: 'success.light' }}>
                  <MonetizationOn color="success" />
                </Avatar>
              </Box>
              <Box sx={{ display: 'flex', alignItems: 'center', gap: 1 }}>
                <Chip
                  icon={<ArrowUpward />}
                  label="+12.5%"
                  size="small"
                  color="success"
                  variant="outlined"
                />
                <Typography variant="caption" color="text.secondary">
                  vs last month
                </Typography>
              </Box>
            </CardContent>
          </Card>
        </Grid>

        <Grid item xs={12} sm={6} md={3}>
          <Card>
            <CardContent>
              <Box sx={{ display: 'flex', justifyContent: 'space-between', mb: 2 }}>
                <Box>
                  <Typography color="text.secondary" variant="body2">
                    Total Views
                  </Typography>
                  <Typography variant="h5" fontWeight="bold">
                    {formatNumber(stats?.totalViews || 0)}
                  </Typography>
                </Box>
                <Avatar sx={{ bgcolor: 'primary.light' }}>
                  <Visibility color="primary" />
                </Avatar>
              </Box>
              <Box sx={{ display: 'flex', alignItems: 'center', gap: 1 }}>
                <Chip
                  icon={<ArrowUpward />}
                  label="+18.2%"
                  size="small"
                  color="primary"
                  variant="outlined"
                />
                <Typography variant="caption" color="text.secondary">
                  vs last month
                </Typography>
              </Box>
            </CardContent>
          </Card>
        </Grid>

        <Grid item xs={12} sm={6} md={3}>
          <Card>
            <CardContent>
              <Box sx={{ display: 'flex', justifyContent: 'space-between', mb: 2 }}>
                <Box>
                  <Typography color="text.secondary" variant="body2">
                    Videos Published
                  </Typography>
                  <Typography variant="h5" fontWeight="bold">
                    {stats?.totalVideos || 0}
                  </Typography>
                </Box>
                <Avatar sx={{ bgcolor: 'secondary.light' }}>
                  <VideoLibrary color="secondary" />
                </Avatar>
              </Box>
              <Box sx={{ display: 'flex', alignItems: 'center', gap: 1 }}>
                <Typography variant="body2" color="text.secondary">
                  {stats?.videosPublishedToday || 0} today
                </Typography>
              </Box>
            </CardContent>
          </Card>
        </Grid>

        <Grid item xs={12} sm={6} md={3}>
          <Card>
            <CardContent>
              <Box sx={{ display: 'flex', justifyContent: 'space-between', mb: 2 }}>
                <Box>
                  <Typography color="text.secondary" variant="body2">
                    Profit Margin
                  </Typography>
                  <Typography variant="h5" fontWeight="bold">
                    {((stats?.profit || 0) / (stats?.totalRevenue || 1) * 100).toFixed(1)}%
                  </Typography>
                </Box>
                <Avatar sx={{ bgcolor: 'warning.light' }}>
                  <TrendingUp color="warning" />
                </Avatar>
              </Box>
              <Box sx={{ display: 'flex', alignItems: 'center', gap: 1 }}>
                <Typography variant="body2" color="success.main">
                  {formatCurrency(stats?.profit || 0)} profit
                </Typography>
              </Box>
            </CardContent>
          </Card>
        </Grid>
      </Grid>

      {/* Charts and Activity */}
      <Grid container spacing={3}>
        {/* Revenue Chart */}
        <Grid item xs={12} md={8}>
          <Paper sx={{ p: 3, height: 400 }}>
            <Box sx={{ display: 'flex', justifyContent: 'space-between', mb: 2 }}>
              <Typography variant="h6" fontWeight="bold">
                Revenue & Cost Analysis
              </Typography>
              <Button
                size="small"
                startIcon={<AnalyticsIcon />}
                onClick={() => navigate('/analytics')}
              >
                View Details
              </Button>
            </Box>
            <Box sx={{ height: 320 }}>
              <Line data={revenueChartData} options={chartOptions} />
            </Box>
          </Paper>
        </Grid>

        {/* Video Queue */}
        <Grid item xs={12} md={4}>
          <Paper sx={{ p: 3, height: 400 }}>
            <Box sx={{ display: 'flex', justifyContent: 'space-between', mb: 2 }}>
              <Typography variant="h6" fontWeight="bold">
                Video Queue ({videosInQueue.length})
              </Typography>
              <Button
                size="small"
                onClick={() => navigate('/videos/queue')}
              >
                View All
              </Button>
            </Box>
            <List sx={{ maxHeight: 320, overflow: 'auto' }}>
              {videosInQueue.length === 0 ? (
                <Box sx={{ textAlign: 'center', py: 4 }}>
                  <Schedule sx={{ fontSize: 48, color: 'text.secondary', mb: 2 }} />
                  <Typography color="text.secondary">
                    No videos in queue
                  </Typography>
                  <Button
                    size="small"
                    startIcon={<Add />}
                    onClick={() => navigate('/videos/generate')}
                    sx={{ mt: 2 }}
                  >
                    Generate Video
                  </Button>
                </Box>
              ) : (
                videosInQueue.map((video) => (
                  <ListItem
                    key={video.id}
                    sx={{
                      border: '1px solid',
                      borderColor: 'divider',
                      borderRadius: 1,
                      mb: 1,
                    }}
                  >
                    <ListItemAvatar>
                      <Avatar src={video.thumbnail} variant="rounded">
                        <VideoLibrary />
                      </Avatar>
                    </ListItemAvatar>
                    <ListItemText
                      primary={video.title}
                      secondary={
                        <Box>
                          <Typography variant="caption" color="text.secondary">
                            {video.channel}
                          </Typography>
                          {video.status === 'generating' && video.progress && (
                            <LinearProgress
                              variant="determinate"
                              value={video.progress}
                              sx={{ mt: 1 }}
                            />
                          )}
                          {video.status === 'scheduled' && (
                            <Chip
                              icon={<Schedule />}
                              label={video.scheduledTime}
                              size="small"
                              sx={{ mt: 1 }}
                            />
                          )}
                        </Box>
                      }
                    />
                  </ListItem>
                ))
              )}
            </List>
          </Paper>
        </Grid>

        {/* Recent Activity */}
        <Grid item xs={12} md={6}>
          <Paper sx={{ p: 3, height: 400 }}>
            <Typography variant="h6" fontWeight="bold" sx={{ mb: 2 }}>
              Recent Activity
            </Typography>
            <List sx={{ maxHeight: 340, overflow: 'auto' }}>
              {recentActivity.map((activity) => (
                <ListItem key={activity.id}>
                  <ListItemAvatar>
                    <Avatar sx={{ bgcolor: activity.color }}>
                      {activity.icon}
                    </Avatar>
                  </ListItemAvatar>
                  <ListItemText
                    primary={activity.title}
                    secondary={
                      <Box>
                        <Typography variant="body2" color="text.secondary">
                          {activity.description}
                        </Typography>
                        <Typography variant="caption" color="text.secondary">
                          {activity.timestamp}
                        </Typography>
                      </Box>
                    }
                  />
                </ListItem>
              ))}
            </List>
          </Paper>
        </Grid>

        {/* Category Performance */}
        <Grid item xs={12} md={6}>
          <Paper sx={{ p: 3, height: 400 }}>
            <Typography variant="h6" fontWeight="bold" sx={{ mb: 2 }}>
              Content Categories
            </Typography>
            <Box sx={{ height: 340, display: 'flex', alignItems: 'center', justifyContent: 'center' }}>
              <Box sx={{ width: 280, height: 280 }}>
                <Doughnut data={categoryChartData} />
              </Box>
            </Box>
          </Paper>
        </Grid>

        {/* Best Performing Video */}
        {stats?.bestPerformingVideo && (
          <Grid item xs={12}>
            <Paper sx={{ p: 3 }}>
              <Box sx={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
                <Box sx={{ display: 'flex', alignItems: 'center', gap: 2 }}>
                  <Avatar sx={{ bgcolor: 'success.main', width: 56, height: 56 }}>
                    <TrendingUp />
                  </Avatar>
                  <Box>
                    <Typography variant="h6" fontWeight="bold">
                      Best Performing Video
                    </Typography>
                    <Typography variant="body1">
                      {stats.bestPerformingVideo.title}
                    </Typography>
                    <Box sx={{ display: 'flex', gap: 3, mt: 1 }}>
                      <Typography variant="body2" color="text.secondary">
                        <strong>{formatNumber(stats.bestPerformingVideo.views)}</strong> views
                      </Typography>
                      <Typography variant="body2" color="text.secondary">
                        <strong>{formatCurrency(stats.bestPerformingVideo.revenue)}</strong> revenue
                      </Typography>
                    </Box>
                  </Box>
                </Box>
                <Button
                  variant="outlined"
                  onClick={() => navigate(`/videos/${stats.bestPerformingVideo?.id}`)}
                >
                  View Details
                </Button>
              </Box>
            </Paper>
          </Grid>
        )}
      </Grid>
    </Box>
  );
};